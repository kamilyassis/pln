from pydantic import BaseModel, Field


class Article(BaseModel):
    category: str = Field(..., description="The category of the article")
    title: str = Field(..., description="The title of the article")
    subtitle: str = Field(..., description="The subtitle of the article")
    url: str = Field(..., description="The URL of the article")


class ArticleContent(BaseModel):
    title: str = Field(..., description="The title of the article")
    category: str = Field(..., description="The category of the article")
    author: str = Field(..., description="The author of the article")
    date: str = Field(..., description="The publication date of the article")
    content: str = Field(..., description="The full content of the article")
    keywords: list[str] = Field(..., description="A list of keywords associated with the article")
    more_links: list[str] = Field(..., description="A list of additional links found in the article")


class MinimalArticleContent(BaseModel):
    id: str = Field(..., description="A unique identifier for the article")
    title: str = Field(..., description="The title of the article")
    content: str = Field(..., description="The full content of the article")