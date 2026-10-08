from ragforge.core.config import get_settings
from ragforge.app.generation.service import answer_query


QUESTIONS = [
    {
        "id": "q01",
        "question": "What is the monthly rent amount?",
        "expected": "$750 per month",
    },
    {
        "id": "q02",
        "question": "When does the lease term start and end?",
        "expected": "Starts August 1, 2026 and ends July 31, 2027",
    },
    {
        "id": "q03",
        "question": "What is the security deposit amount?",
        "expected": "$750",
    },
    {
        "id": "q04",
        "question": "What are all the conditions under which the landlord can enter the premises?",
        "expected": (
            "With reasonable prior notice (typically 24 hours) for "
            "inspection, maintenance, or repair between 9AM and 8PM. "
            "Without notice in a bona fide emergency such as fire, "
            "flood, or gas leak. During the last 60 days of the lease "
            "to show the premises to prospective tenants or buyers."
        ),
    },
    {
        "id": "q05",
        "question": "What is the tenant responsible for paying regarding utilities?",
        "expected": (
            "70% of all utility bills including water, electricity, "
            "wifi, and snow removal. Also responsible for a $60 "
            "additional charge if any utility is turned off and "
            "reverts to the landlord."
        ),
    },
    {
        "id": "q06",
        "question": "If the tenant pays rent on the 6th of the month, what is the total amount owed?",
        "expected": "$800 — $750 rent plus $50 late fee",
    },
    {
        "id": "q07",
        "question": "Can the tenant sublet the apartment to a friend for two months while traveling?",
        "expected": (
            "No. Subletting requires prior written consent from the "
            "landlord."
        ),
    },
    {
        "id": "q08",
        "question": "What is the landlord's phone number?",
        "expected": "NOT IN DOCUMENT",
    },
    {
        "id": "q09",
        "question": "What is the penalty for breaking the lease early?",
        "expected": "NOT IN DOCUMENT",
    },
    {
        "id": "q10",
        "question": "What must the tenant do before the lease expires to avoid automatic termination?",
        "expected": (
            "Provide written notice to the landlord no later than "
            "60 days before the lease expiration date stating intent "
            "to either renew or vacate."
        ),
    },
]


def main():
    settings = get_settings()

    for item in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"{item['id']}: {item['question']}")
        print("=" * 80)

        response = answer_query(
            settings=settings,
            query=item["question"],
            top_k=5,
        )

        print("\nANSWER:")
        print(response.answer)

        print("\nEXPECTED:")
        print(item["expected"])

        print("\nSOURCES:")

        for i, source in enumerate(response.sources, start=1):
            print(
                f"[Source {i}] "
                f"score={source.score:.4f} "
                f"page={source.metadata.get('page_number')} "
                f"section={source.metadata.get('section_number')}"
            )


if __name__ == "__main__":
    main()