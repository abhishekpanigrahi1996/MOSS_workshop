import { useState } from "react";

const Card = ({ children }) => (
  <div className="border rounded-2xl shadow p-4 bg-white">{children}</div>
);

const CardContent = ({ children }) => <div>{children}</div>;

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
            <CardContent>
              <a href={paper.url} target="_blank" rel="noopener noreferrer" className="text-xl font-semibold hover:underline">
                {paper.title}
              </a>
              <p className="text-sm text-gray-600">Authors: {paper.authors.join(", ")}</p>
              <p className="text-sm text-gray-500 italic mb-2">{paper.tldr}</p>
              <p className="text-sm text-gray-800 mb-1">{paper.abstract}</p>
              <p className="text-xs text-gray-500">Keywords: {paper.keywords.join(", ")}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}