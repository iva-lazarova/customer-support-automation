import os
import sys
import traceback
import langchain_community
import warnings
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool

# Suppress warnings
warnings.filterwarnings("ignore")

# Debug: Print environment variables
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("SERPER_API_KEY:", os.getenv("SERPER_API_KEY"))

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

try:
    print("Defining agents...")
    # Define Agents
    support_agent = Agent(
        role="Support Representative",
        goal="Be the most friendly and helpful support representative in your team",
        backstory=(
            "You work in the Risk Services Practice at PwC Singapore (https://www.pwc.com/sg/en.html) and "
            "are now working on providing support to {customer}, for your company. "
            "You need to make sure that you provide the best support!"
            "Make sure to provide full complete answers, and make no assumptions."
        ),
        allow_delegation=False,
        verbose=True
    )

    support_quality_assurance_agent = Agent(
        role="Support Quality Assurance Specialist",
        goal="Get recognition for providing the best support quality assurance in your team",
        backstory=(
            "You work in the Risk Services Practice at PwC Singapore (https://www.pwc.com/sg/en.html) and "
            "are now working with your team on a request from {customer} ensuring that the support representative is "
            "providing the best support possible.\n"
            "You need to make sure that the support representative is providing full"
            "complete answers, and makes no assumptions."
        ),
        verbose=True
    )

    print("Defining tools...")
    # Tools
    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()
    docs_scrape_tool = ScrapeWebsiteTool(
        website_url="https://www.pwc.com/sg/en/services/risk/compliance.html"
    )

    print("Creating tasks...")
    # Create the Tasks
    inquiry_resolution = Task(
        description=(
            "{customer} just reached out with a super important ask:\n"
            "{inquiry}\n\n"
            "{person} from {customer} is the one that reached out. "
            "Make sure to use everything you know to provide the best support possible."
            "You must strive to provide a complete and accurate response to the customer's inquiry."
        ),
        expected_output=(
            "A detailed, informative response to the customer's inquiry that addresses "
            "all aspects of their question.\n"
            "The response should include references to everything you used to find the answer, "
            "including external data or solutions. Ensure the answer is complete, "
            "leaving no questions unanswered, and maintain a helpful and friendly "
            "tone throughout."
        ),
        tools=[docs_scrape_tool],
        agent=support_agent
    )

    quality_assurance_review = Task(
        description=(
            "Review the response drafted by the Support Representative for {customer}'s inquiry. "
            "Ensure that the answer is comprehensive, accurate, and adheres to the "
            "high-quality standards expected for customer support.\n"
            "Verify that all parts of the customer's inquiry have been addressed "
            "thoroughly, with a helpful and friendly tone.\n"
            "Check for references and sources used to find the information, "
            "ensuring the response is well-supported and leaves no questions unanswered."
        ),
        expected_output=(
            "A final, detailed, and informative response ready to be sent to the customer.\n"
            "This response should fully address the customer's inquiry, incorporating all "
            "relevant feedback and improvements.\n"
            "Don't be too formal, but maintain a professional and friendly tone."
        ),
        agent=support_quality_assurance_agent,
    )

    print("Assembling the crew...")
    # Assemble the Crew
    crew = Crew(
        agents=[support_agent, support_quality_assurance_agent],
        tasks=[inquiry_resolution, quality_assurance_review],
        verbose=2,
        memory=True
    )

    print("Crew assembled successfully.")

    # Main execution block
    if __name__ == "__main__":
        print("Starting main execution...")
        inputs = {
            "customer": "Trade Fiction Company",
            "person": "Beagel Thomanson",
            "inquiry": "I am planning to apply for an IPO listing in Singapore. "
                       "Can you explain to me what is SOX/ J-SOX compliance, "
                       "whether I need to implement it since I will only list "
                       "in Singapore and have no operations in US or Japan, "
                       "what the benefits will be to me and who from your staff can help? "
                       "Can you provide guidance?"
        }
        print("Kicking off the crew...")
        result = crew.kickoff(inputs=inputs)
        print("Crew execution completed.")
        print("Result:", result)

except Exception as e:
    print(f"An error occurred: {e}")
    print("Traceback:")
    traceback.print_exc()

print("Script execution completed.")