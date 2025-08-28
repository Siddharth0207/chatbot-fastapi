from langchain.prompts import ChatPromptTemplate, PromptTemplate


# Extraction prompt used by the LLM to parse user input into structured fields
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that extracts diamond preferences into structured fields."),
    ("human", "Extract the diamond query fields from: {input}\n\n{format_instructions}")
])


# Summary prompt used to generate a user-facing summary of retrieved diamonds
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["diamonds", "user_query", "total_count"],
    template="""
    You are an expert gemologist and diamond consultant.

    The user asked: "{user_query}"

    You have retrieved the following diamonds from the database:

    {diamonds}

    Now do the following:

    1. Begin with this line exactly:
    We found total {total_count} stones based on your query.

    2. Then say:
    Here we are displaying top 10 stones:

    3. Print each diamond on a **new line**, following this format:
    . Shape: <Shape>, Color: <Color>, Cut: <Cut>, Clarity: <Clarity>, Polish: <Polish>, Weight: <Carat>, Price/Carat: <$Price>

    4. Ensure that each diamond is printed on a separate line with a line break (`\n`).

    5. After listing the diamonds, provide a **2-3 sentence summary** of what makes this selection valuable and how it fits the user’s preferences.

    Make sure the response preserves the line breaks exactly as instructed, so it renders properly in a web frontend.
    """
)
