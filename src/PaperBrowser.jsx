import { useState } from "react";

const Card = ({ children }) => (
  <div className="border rounded-2xl shadow p-4 bg-white">{children}</div>
);

const Input = ({ className = "", ...props }) => (
  <input
    {...props}
    className={`border p-2 rounded w-full ${className}`.trim()}
  />
);

const papers = [
  {
    id: 96,
    title: "Understanding Attention Glitches with Threshold Relative Attention",
    authors: ["Mattia Opper", "Roland Fernandez", "Paul Smolensky", "Jianfeng Gao"],
    keywords: ["length generalisation", "attention glitches", "flip-flops", "algorithmic reasoning"],
    tldr: "We create a novel attention mechanism to address some limitations of standard self-attention and apply it to the flip-flop language modeling task",
    abstract: "Transformers struggle with generalisation, displaying poor performance even on basic yet fundamental tasks, such as flip-flop language modeling...",
    url: "https://openreview.net/forum?id=yhNOZsCPUi"
  },
  // Add the rest of the paper objects here
];

export default function PaperBrowser() {
  const [query, setQuery] = useState("");
  const [openAbstract, setOpenAbstract] = useState(null);

  const filtered = papers.filter(paper =>
    paper.title.toLowerCase().includes(query.toLowerCase()) ||
    paper.authors.some(a => a.toLowerCase().includes(query.toLowerCase())) ||
    paper.keywords.some(k => k.toLowerCase().includes(query.toLowerCase()))
  );

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">MOSS 2025 Accepted Papers</h1>
      <Input
        placeholder="Search by title, author, keyword..."
        className="mb-6"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="grid gap-4">
        {filtered.map(paper => (
          <Card key={paper.id}>
            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-start">
                <div className="w-full">
                  <a
                    href={paper.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-lg font-semibold text-blue-600 hover:underline"
                  >
                    {paper.title}
                  </a>
                  <p className="text-sm text-gray-700">{paper.authors.join(", ")}</p>
                  <p className="text-sm text-gray-600 italic">{paper.tldr}</p>
                  <p className="text-xs text-gray-500">Keywords: {paper.keywords.join(", ")}</p>
                </div>
                <div className="ml-4">
                  <button
                    onClick={() => setOpenAbstract(openAbstract === paper.id ? null : paper.id)}
                    className="text-sm text-blue-500 hover:underline whitespace-nowrap"
                  >
                    {openAbstract === paper.id ? "Hide Abstract" : "Show Abstract"}
                  </button>
                </div>
              </div>
              {openAbstract === paper.id && (
                <p className="text-sm text-gray-800 border-t pt-2">{paper.abstract}</p>
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
