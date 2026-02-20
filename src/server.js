const express = require("express");
const { createClient } = require("@supabase/supabase-js");
const { v4: uuidv4 } = require("uuid");
const { GoogleGenAI } = require("@google/genai");
const { chunkText } = require("./utils");
require("dotenv").config();

const app = express();
app.use(express.json());

app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE");
  res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.header("Access-Control-Allow-Credentials", "true");

  if (req.method === "OPTIONS") {
    return res.sendStatus(200);
  }

  next();
});

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_KEY,
);

// generate UUID anonym per session (simple demo)
const userId = uuidv4();

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.post("/query", async (req, res) => {
  const { question, user_id } = req.body;
  try {
    // 1️⃣ Embed the question
    const embeddingResp = await ai.models.embedContent({
      model: "gemini-embedding-001",
      contents: question,
      config: {
        outputDimensionality: 768,
      },
    });
    const queryVector = embeddingResp.embeddings[0].values;

    // 2️⃣ Query Supabase using pgvector distance
    const { data: chunks, error } = await supabase.rpc("match_chunks", {
      query_embedding: queryVector,
      match_count: 5,
      filter_user_id: user_id ?? null,
    });
    if (error) throw error;

    if (!chunks.length) return res.json({ answer: "No documents found." });

    const context = chunks.map((c) => c.content).join("\n\n");

    // 3️⃣ Set SSE headers for streaming
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    // 4️⃣ Stream from Gemini using generateContentStream
    const stream = await ai.models.generateContentStream({
      model: "gemini-2.5-flash",
      contents: `You are a helpful AI assistant.
Answer strictly based on the provided context.
If the answer is not in the context, say you don't know.

Context:
${context}

Question: ${question}`,
    });

    // 5️⃣ Pipe each chunk to the client
    for await (const chunk of stream) {
      const text = chunk.text;
      if (text) {
        res.write(`data: ${JSON.stringify({ text })}\n\n`);
      }
    }

    // 6️⃣ Send retrieved chunks and close the stream
    res.write(
      `data: ${JSON.stringify({ retrieved_chunks: chunks, done: true })}\n\n`,
    );
    res.end();
  } catch (err) {
    console.error(err);
    // If headers already sent, close the stream with an error event
    if (res.headersSent) {
      res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    } else {
      res.status(500).json({ error: err.message });
    }
  }
});

// Accept single or array
app.post("/documents", async (req, res) => {
  const docs = Array.isArray(req.body) ? req.body : [req.body]; // backward-compatible

  const rows = docs.map(({ title, content }) => ({
    title,
    content,
    user_id: userId,
  }));
  const { data, error } = await supabase
    .from("documents")
    .insert(rows)
    .select();
  if (error) return res.status(500).json({ error: error.message });
  for (d of data) {
    try {
      const chunks = chunkText(d.content, 300); // chunk per 300 kata

      const insertedChunks = [];

      const embeddingResp = await ai.models.embedContent({
        model: "gemini-embedding-001",
        contents: chunks,
        config: {
          outputDimensionality: 768, // ← must be inside config
        },
      });

      const vectors = embeddingResp.embeddings.map((e) => e.values);

      for (let i = 0; i < chunks.length; i++) {
        // 3️⃣ Save to chunks table
        const { data, error } = await supabase
          .from("chunks")
          .insert([
            {
              user_id: d.user_id ?? null,
              document_id: d.id,
              content: chunks[i],
              embedding: vectors[i],
            },
          ])
          .select();

        if (error) throw error;
        insertedChunks.push(data[0]);
      }
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: err.message });
    }
  }
  res.json({ message: "Documents saved", data });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
