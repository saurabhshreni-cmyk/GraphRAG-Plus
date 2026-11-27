// Example questions shown under the Ask box, tailored to the open corpus.
//
// Generic examples ("What is an adjacency matrix?") were unrelated to the
// demo documents, so clicking one only ever produced a refusal. The demo gets
// questions that show off the system (a multi-hop link, a contradiction);
// any other corpus gets questions built from its most-connected entities.

export const DEMO_CORPUS_ID = "corpus_demo_nova";

const DEMO_SUGGESTIONS = [
  "What did Nova Dynamics acquire?",
  "Was Project Helios cancelled?",
  "How is Orion Labs connected to Nova Dynamics?",
];

const MAX_SUGGESTIONS = 3;

/**
 * @param {{corpus_id: string}|null} corpus  the active corpus
 * @param {{nodes?: Array<{label?: string, node_type?: string, degree?: number}>}} snapshot
 * @returns {string[]}
 */
export function buildSuggestions(corpus, snapshot) {
  if (!corpus) return [];
  if (corpus.corpus_id === DEMO_CORPUS_ID) return DEMO_SUGGESTIONS;

  const entities = (snapshot?.nodes || [])
    .filter((n) => n.label && !["Document", "Chunk"].includes(n.node_type))
    .sort((a, b) => (b.degree || 0) - (a.degree || 0))
    .map((n) => String(n.label).trim())
    .filter((label, i, all) => label.length > 1 && all.indexOf(label) === i);

  const [first, second, third] = entities;
  const questions = [];
  if (first) questions.push(`What do the documents say about ${first}?`);
  if (first && second) questions.push(`How is ${first} related to ${second}?`);
  if (third) questions.push(`What is ${third}?`);
  return questions.slice(0, MAX_SUGGESTIONS);
}
