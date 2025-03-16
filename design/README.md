## Architecture diagram

Currently under development at: [excalidraw](https://excalidraw.com/#room=a5414cbf87bf1607b938,Q8yx29hghtOyOZcZv6n3pw)

## PlantUML

You need to install PlantUML to be able to edit and regenerate the uml diagrams. You can find how to install plantuml on the internet.

Assuming you're currently in this directory (i.e. `design/`), the command I run to generate a `.png` from a `.puml` file is:

```zsh
plantuml -o images/ <filename>.puml
```

## A guide for consistent nomenclature:

I am currently using these terms to mean these things in my design documents:

- BingeBuddy: The agent that is going to interact with the user via the front-end
- SemanticAgentState/SemanticWorkflow: The semantic agent here is not our conversational agent. This is our longterm
  memory agent that aggregates memories. Essentially, everything that has "semantic" refers to the memory agent that we want to
  build according to our report.
- EpisodicAgentState/EpisodicWorkflow: The word "episodic" refers to the memory agent we want to test against as mentioned in our report.

For e.g. the SemanticAgentState class has an additional attribute called "aggregated_memories" while the EpisodicAgentState does not;
it only uses "extracted_memories".
