const express = require("express");
const { v4: uuidv4 } = require("uuid");
const { GoogleGenAI } = require("@google/genai");
const { chunkText } = require("./utils");
const cors = require("cors");
const pool = require("./db"); // your pg pool
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

const userId = uuidv4();

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.use(
	cors({
		origin: ["https://azizjabbar.tech"],
	}),
);

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

		// 2️⃣ Query using pgvector similarity search (replicates supabase.rpc("match_chunks"))
		const vectorStr = `[${queryVector.join(",")}]`;
		const { rows: chunks } = await pool.query(
			`SELECT id, content, document_id, user_id,
                    1 - (embedding <=> $1::vector) AS similarity
             FROM chunks
             WHERE ($2::uuid IS NULL OR user_id = $2::uuid)
             ORDER BY embedding <=> $1::vector
             LIMIT 5`,
			[vectorStr, user_id ?? null],
		);

		if (!chunks.length) return res.json({ answer: "No documents found." });

		const context = chunks.map((c) => c.content).join("\n\n");

		// 3️⃣ Set SSE headers for streaming
		res.setHeader("Content-Type", "text/event-stream");
		res.setHeader("Cache-Control", "no-cache");
		res.setHeader("Connection", "keep-alive");

		// 4️⃣ Stream from Gemini
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
		if (res.headersSent) {
			res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
			res.end();
		} else {
			res.status(500).json({ error: err.message });
		}
	}
});

app.post("/documents", async (req, res) => {
	const docs = Array.isArray(req.body) ? req.body : [req.body];

	const client = await pool.connect();
	try {
		await client.query("BEGIN");

		const insertedDocs = [];
		for (const { title, content } of docs) {
			const { rows } = await client.query(
				`INSERT INTO documents (title, content, user_id)
                 VALUES ($1, $2, $3)
                 RETURNING *`,
				[title, content, userId],
			);
			insertedDocs.push(rows[0]);
		}

		// Embed and insert chunks for each document
		for (const d of insertedDocs) {
			const chunks = chunkText(d.content, 300);

			const embeddingResp = await ai.models.embedContent({
				model: "gemini-embedding-001",
				contents: chunks,
				config: {
					outputDimensionality: 768,
				},
			});

			const vectors = embeddingResp.embeddings.map((e) => e.values);

			for (let i = 0; i < chunks.length; i++) {
				const vectorStr = `[${vectors[i].join(",")}]`;
				await client.query(
					`INSERT INTO chunks (user_id, document_id, content, embedding)
                     VALUES ($1, $2, $3, $4::vector)`,
					[d.user_id ?? null, d.id, chunks[i], vectorStr],
				);
			}
		}

		await client.query("COMMIT");
		res.json({ message: "Documents saved", data: insertedDocs });
	} catch (err) {
		await client.query("ROLLBACK");
		console.error(err);
		res.status(500).json({ error: err.message });
	} finally {
		client.release();
	}
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
