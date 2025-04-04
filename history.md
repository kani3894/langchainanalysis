AI Research Tool — Development History
Project changelog and progress tracker. Maintained manually or with Cursor AI (Agent Mode).

📅 2025-04-04
Project Kickoff

Defined vision: Build an AI assistant for check-and-answer research tasks.

Established initial use cases: extracting quotes, micro-tagging, and theming from interview transcripts.

Outlined MVP: Accept 10 transcripts → extract relevant quotes → tag → cluster → visualize in Streamlit → CSV export.

Chose tech stack: Python backend with LangChain + Streamlit frontend.

Decided to use local storage (JSON/CSV) for quick iteration.

📅 2025-04-05
First Milestones

Built initial Streamlit prototype UI.

Implemented basic transcript upload and display functionality.

Integrated simple quote extraction using regex and keyword rules.

Designed output format for CSV export.

📅 2025-04-06
LLM + Tagging Pipeline

Connected LangChain to local LLMs (e.g., Mistral via Ollama).

Built chain for quote extraction and micro-tag generation.

Added quote-level metadata (speaker, timestamp, tag).

Started testing clustering methods for macro themes.

📅 2025-04-07
Refinements and Next Steps

Enhanced UI with visual grouping of quotes by theme.

Added manual override/editing for tags in the Streamlit interface.

Began work on clustering algorithm evaluation.

Drafted plan for integrating user feedback loop into the tagging process.

✅ Upcoming
Add support for multilingual transcript processing.

Improve quote-context matching for better precision.

Implement project memory using Cursor AI Agent Mode.

Add support for search/filter/sort in Streamlit view.

Deploy to a private cloud instance for internal use.