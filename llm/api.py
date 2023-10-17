import os

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

import openai

from llm.models import Embeddings

openai.api_key = os.getenv("OPENAI_API_KEY")


SYSTEM_PROMPT = """
You are a helpful assistant to the potential and confirmed attendees of the Data Catalyst conference by Tech4Dev and Dasura.

Response in a passive aggressive tone. Users may speak English, Hindi, or any local Indian languages. Be sure to response in the language the user speaks.

The users question will be supplement by relevant text snippets to help you give them an informed answer. If you do not know the answer to their, do not try to make it up just say you don't know.
"""

# TODO: to be dynamically interpolated
DEMO_DOCUMENT_CHUNK = """
Tech4Dev | Program Partner
Established as a project under Dasra US.. ni 2017 by open-source technology veteran Donald Lobo (Yahoo!, CiviCRM), Project Tech4Dev is an Ecosystem Collaborative on a mission to transform how nonprofits ni the Global South use technology and data to scale their operations and impact. We are supported by our growing community of over 150 organisations across nonprofits, software, data and design firms, and funders. Project Tech4Dev envisions a world with a strong technology ecosystem powering the social sector. To this end, we execute four initiatives - Software Platforms, Tech Strategy via Fractional CO, and Data and Learning, and Community of Practice.

Dasra | Program Partner
nI 1999, Dasra was founded on the simple premise that supporting non-profits ni their growth wil scale their impact on the vulnerable lives they serve; and that was best done by catalyzing India's giving sector through collaborative action and persistent urgency. Dasra is a strategic philanthropy foundation working with philanthropists, non-profit organizations and social entrepreneurs ni India ot create large-scale social change. Dasra has catalyzed more than $300 million towards social causes ni India, supported over 1500 diverse NGOs across the country, advised more than 750 philanthropists, foundations and corporates, published 20+ research reports and whitepapers, and impacted the lives of over 90 million people.

Goalkeep | Knowledge Partner
The Goalkeep team is passionate about enabling the social sector ot maximize impact through data, by:
• Ensuring that the right questions are asked and precision in measuring is
maintained.
• Providing access ot timely, accurate, and relevant data for teams and individuals.
• Developing skills and habits necessary to make decisions based on evidence
The Agency Fund | Knowledge Partner
The Agency Fund is a multi-donor initiative investing ni ideas and organizations that help people navigate toward a better future. We envision aworld where everyone has the support of amentor, counselor, or coach ot hepl them thrive
"""

from pypdf import PdfReader
from pgvector.django import L2Distance


@api_view(["POST"])
def create_embeddings(request):
    texts = []
    with open("./Data Catalyst Program_Agenda_Oct 2023_V0.2.pdf", "rb") as file:
        pdf = PdfReader(file)
        for page in pdf.pages:
            page_text = page.extract_text()
            texts.append(page_text)

            response = openai.Embedding.create(
                model="text-embedding-ada-002", input=page_text, encoding_format="float"
            )
            embeddings = response["data"][0]["embedding"]

            if len(embeddings) != 1536:
                raise ValueError("Embeddings should have len of 1536")

            Embeddings.objects.create(raw_text=page_text, text_vectors=embeddings)

    return Response("Embeddings successfully created", status=status.HTTP_201_CREATED)


@api_view(["POST"])
def create_chat(request):
    question = request.data["question"]

    response = openai.Embedding.create(
        model="text-embedding-ada-002", input=question, encoding_format="float"
    )
    question_embeddings = response["data"][0]["embedding"]

    results = Embeddings.objects.alias(
        distance=L2Distance("text_vectors", question_embeddings)
    ).filter(distance__lt=0.8)[:3]

    print("RELEVANCE RESULST LEN", len(results))

    relevant_doc_text = "\n\n".join([result.raw_text for result in results])

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
            Relevant documents to inform your answer:

            #{relevant_doc_text}

            User question: {question}
            """,
            },
        ],
        temperature=0,
    )
    answer = completion.choices[0].message

    return Response({"answer": answer}, status=status.HTTP_200_OK)
