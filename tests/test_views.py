import pytest
import sqlalchemy as sa
import sqlalchemy.orm
from unittest import mock

from sqlalchemy_utils import (
    create_materialized_view,
    create_view,
    refresh_materialized_view
)
from sqlalchemy_utils.view import CreateView, DropView, RefreshMaterializedView


@pytest.fixture
def Article(Base, User):
    class Article(Base):
        __tablename__ = 'article'
        id = sa.Column(sa.Integer, primary_key=True)
        name = sa.Column(sa.String(128))
        author_id = sa.Column(sa.Integer, sa.ForeignKey(User.id))
        author = sa.orm.relationship(User)
    return Article


@pytest.fixture
def User(Base):
    class User(Base):
        __tablename__ = 'user'
        id = sa.Column(sa.Integer, primary_key=True)
        name = sa.Column(sa.String(128))
    return User


@pytest.fixture
def ArticleMV(Base, Article, User):
    class ArticleMV(Base):
        __table__ = create_materialized_view(
            name='article-mv',
            selectable=sa.select(
                Article.id,
                Article.name,
                User.id.label('author_id'),
                User.name.label('author_name'),
            ).select_from(
                Article.__table__.join(User, Article.author_id == User.id)
            ),
            aliases={'name': 'article_name'},
            metadata=Base.metadata,
            indexes=[sa.Index('article-mv_id_idx', 'id')]
        )
    return ArticleMV


@pytest.fixture
def ArticleView(Base, Article, User):
    class ArticleView(Base):
        __table__ = create_view(
            name='article-view',
            selectable=sa.select(
                Article.id,
                Article.name,
                User.id.label('author_id'),
                User.name.label('author_name'),
            ).select_from(
                Article.__table__.join(User, Article.author_id == User.id)
            ),
            metadata=Base.metadata
        )
    return ArticleView


@pytest.fixture
def init_models(ArticleMV, ArticleView):
    pass


@pytest.mark.usefixtures('postgresql_dsn')
class TestMaterializedViews:
    def test_refresh_materialized_view(
        self,
        session,
        Article,
        User,
        ArticleMV
    ):
        article = Article(
            name='Some article',
            author=User(name='Some user')
        )
        session.add(article)
        session.commit()
        refresh_materialized_view(session, 'article-mv')
        materialized = session.query(ArticleMV).first()
        assert materialized.article_name == 'Some article'
        assert materialized.author_name == 'Some user'

    def test_querying_view(
        self,
        session,
        Article,
        User,
        ArticleView
    ):
        article = Article(
            name='Some article',
            author=User(name='Some user')
        )
        session.add(article)
        session.commit()
        row = session.query(ArticleView).first()
        assert row.name == 'Some article'
        assert row.author_name == 'Some user'


class TrivialViewTestCases:
    def life_cycle(
        self,
        engine,
        metadata,
        column,
        cascade_on_drop,
        replace=False,
    ):
        create_view(
            name='trivial_view',
            selectable=sa.select(column),
            metadata=metadata,
            cascade_on_drop=cascade_on_drop,
            replace=replace,
        )
        metadata.create_all(engine)
        metadata.drop_all(engine)


class SupportsCascade(TrivialViewTestCases):
    def test_life_cycle_cascade(
        self,
        connection,
        engine,
        Base,
        User
    ):
        self.life_cycle(engine, Base.metadata, User.id, cascade_on_drop=True)


class DoesntSupportCascade(SupportsCascade):
    @pytest.mark.xfail
    def test_life_cycle_cascade(self, *args, **kwargs):
        super().test_life_cycle_cascade(
            *args,
            **kwargs
        )


class SupportsNoCascade(TrivialViewTestCases):
    def test_life_cycle_no_cascade(
        self,
        connection,
        engine,
        Base,
        User
    ):
        self.life_cycle(engine, Base.metadata, User.id, cascade_on_drop=False)


class SupportsReplace(TrivialViewTestCases):
    def test_life_cycle_replace(
        self,
        connection,
        engine,
        Base,
        User
    ):
        self.life_cycle(
            engine,
            Base.metadata,
            User.id,
            cascade_on_drop=False,
            replace=True,
        )

    def test_life_cycle_replace_existing(
        self,
        connection,
        engine,
        Base,
        User
    ):
        create_view(
            name='trivial_view',
            selectable=sa.select(User.id),
            metadata=Base.metadata,
        )
        Base.metadata.create_all(engine)
        view = CreateView(
            name='trivial_view',
            selectable=sa.select(User.id),
            replace=True,
        )
        with connection.begin():
            connection.execute(view)
        Base.metadata.drop_all(engine)

    def test_replace_materialized(
        self,
        connection,
        engine,
        Base,
        User
    ):
        with pytest.raises(ValueError):
            CreateView(
                name='trivial_view',
                selectable=sa.select(User.id),
                materialized=True,
                replace=True,
            )


@pytest.mark.usefixtures('postgresql_dsn')
class TestPostgresTrivialView(SupportsCascade, SupportsNoCascade, SupportsReplace):
    pass


@pytest.mark.usefixtures('mysql_dsn')
class TestMySqlTrivialView(SupportsCascade, SupportsNoCascade, SupportsReplace):
    pass


@pytest.mark.usefixtures('sqlite_none_database_dsn')
class TestSqliteTrivialView(DoesntSupportCascade, SupportsNoCascade):
    pass


class TestPositionalCompat:
    """Pre-existing public API params must remain positional (backward compat).

    The schema parameter is new (added after these signatures first shipped)
    and must remain keyword-only via a ``*`` separator.
    """

    def test_create_view_positional(self):
        cv = CreateView("v", sa.select(sa.text("1")))
        assert cv.name == "v"
        assert cv.materialized is False
        assert cv.replace is False
        assert cv.schema is None

    def test_create_view_positional_all_params(self):
        cv = CreateView("v", sa.select(sa.text("1")), False, True)
        assert cv.materialized is False
        assert cv.replace is True

    def test_create_view_schema_keyword_only(self):
        with pytest.raises(TypeError):
            CreateView("v", sa.select(sa.text("1")), False, False, "myschema")  # noqa

    def test_drop_view_positional(self):
        dv = DropView("v")
        assert dv.name == "v"
        assert dv.materialized is False
        assert dv.cascade is True
        assert dv.schema is None

    def test_drop_view_positional_all_params(self):
        dv = DropView("v", True, False)
        assert dv.materialized is True
        assert dv.cascade is False

    def test_drop_view_schema_keyword_only(self):
        with pytest.raises(TypeError):
            DropView("v", False, True, "myschema")  # noqa: positional schema

    def test_refresh_materialized_view_positional(self):
        rmv = RefreshMaterializedView("v")
        assert rmv.name == "v"
        assert rmv.concurrently is False
        assert rmv.schema is None

    def test_refresh_materialized_view_concurrently_keyword_only(self):
        rmv = RefreshMaterializedView("v", concurrently=True)
        assert rmv.concurrently is True

    def test_refresh_materialized_view_concurrently_positional_rejected(self):
        with pytest.raises(TypeError):
            RefreshMaterializedView("v", True)  # noqa: positional concurrently

    def test_refresh_materialized_view_schema_keyword_only(self):
        with pytest.raises(TypeError):
            RefreshMaterializedView("v", False, "myschema")  # noqa: positional schema

    def test_create_view_fn_positional(self):
        metadata = sa.MetaData()
        table = create_view("v", sa.select(sa.text("1")), metadata)
        assert table.name == "v"

    def test_create_view_fn_schema_keyword_only(self):
        metadata = sa.MetaData()
        with pytest.raises(TypeError):
            create_view("v", sa.select(sa.text("1")), metadata, True, False, "myschema")  # noqa

    def test_create_materialized_view_fn_positional(self):
        metadata = sa.MetaData()
        table = create_materialized_view(
            "mv", sa.select(sa.text("1")), metadata, None, None
        )
        assert table.name == "mv"

    def test_create_materialized_view_fn_schema_keyword_only(self):
        metadata = sa.MetaData()
        with pytest.raises(TypeError):
            create_materialized_view(
                "mv", sa.select(sa.text("1")), metadata, None, None, True, "myschema"
            )  # noqa: positional schema

    def test_refresh_materialized_view_fn_positional(self):
        session = mock.MagicMock()
        refresh_materialized_view(session, "mv")
        assert session.execute.call_count == 1

    def test_refresh_materialized_view_fn_schema_keyword_only(self):
        session = mock.MagicMock()
        with pytest.raises(TypeError):
            refresh_materialized_view(session, "mv", False, "myschema")  # noqa
        with pytest.raises(TypeError):
            refresh_materialized_view(session, "mv", False, schema="myschema")  # noqa: positional concurrently
