"""Public archive reads behind SceneRecall's local-session authorization."""
from pydantic import Field

from .models import AnimeProfileSelection, Contract


class BangumiSearch(Contract):
    query: str = Field(min_length=1, max_length=200)


class BangumiSubject(Contract):
    reference: str = Field(min_length=1, max_length=500)


def register_bangumi_routes(app, library, search):
    from .anime_profiles import AnimeProfileStore
    from .anime_credits import AnimeCreditsStore
    app.state.bangumi = AnimeProfileStore(library)
    app.state.anime_credits = AnimeCreditsStore(library, app.state.bangumi.client)

    @app.get("/api/assets/{asset_id}/credits")
    def asset_credits(asset_id: str):
        return app.state.anime_credits.get(asset_id)

    @app.post("/api/assets/{asset_id}/credits")
    async def fetch_asset_credits(asset_id: str):
        return await app.state.anime_credits.fetch(asset_id)

    @app.post("/api/bangumi/search")
    async def search_subject(payload: BangumiSearch):
        query = payload.query.strip()
        if not query:
            raise ValueError("请填写动画名称")
        return await app.state.bangumi.search(query)

    @app.post("/api/bangumi/subject")
    async def subject(payload: BangumiSubject):
        return await app.state.bangumi.dossier(payload.reference)

    @app.post("/api/projects/{project_id}/anime-profile")
    async def project_profile(project_id: str, payload: AnimeProfileSelection):
        # Validate the local destination before requesting remote data.
        project = library.organization.get_project(project_id)
        if project["kind"] not in {"animation", "series"}:
            raise ValueError("只有动画项目可以使用 Bangumi 初始档案")
        from .anime_profiles import check_binding
        selection = payload.model_dump(exclude_none=True)
        check_binding(project.get("anime_profile"), selection)
        dossier = await app.state.bangumi.dossier(payload.subject_id) if payload.mode == "bangumi" else None
        app.state.bangumi.apply_project(project_id, selection, dossier)
        await search.rebuild(library.records())
        return library.organization.get_project(project_id)

    @app.post("/api/assets/{asset_id}/anime-profile")
    async def asset_profile(asset_id: str, payload: AnimeProfileSelection):
        asset = library.get_asset(asset_id)
        if asset["kind"] != "animation":
            raise ValueError("只有动画可以使用 Bangumi 初始档案")
        from .anime_profiles import check_binding
        selection = payload.model_dump(exclude_none=True)
        check_binding(asset.get("anime_profile"), selection)
        dossier = await app.state.bangumi.dossier(payload.subject_id) if payload.mode == "bangumi" else None
        result = app.state.bangumi.apply_asset(asset_id, selection, dossier)
        await search.rebuild(library.records())
        return result
