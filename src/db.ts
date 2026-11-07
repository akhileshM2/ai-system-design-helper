import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { openai, supabase } from './config';
import { processAllDocs } from "./docProcessor";

async function splitText(text: string) {
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 250,
        chunkOverlap: 35
    });

    return splitter.createDocuments([text]);
}

async function createAndStoreEmbeddings(folderPath: string) {
    try {
        const documents = await processAllDocs(folderPath);
        console.log(`Extracted ${documents.length} documents from ${folderPath}`);

        for (const doc of documents) {
            if (!doc.content.trim()) {
                console.warn(`⚠️ Skipping ${doc.filename} — no extractable text`);
                continue;
            }

            const chunks = await splitText(doc.content);
            console.log(`Split ${doc.filename} into ${chunks.length} chunks`);

            const data = await Promise.all(
                chunks.map(async (chunk) => {
                    const embeddingResponse = await openai.embeddings.create({
                        model: "text-embedding-ada-002",
                        input: chunk.pageContent
                    });

                    return {
                        content: chunk.pageContent,
                        embedding: embeddingResponse.data[0].embedding
                    };
                })
            );

            await insertInBatches(data)

            console.log(`✅ Stored embeddings for ${doc.filename}`);
        }

        console.log("Embedding and storing complete!!");
    } catch (err: any) {
        console.error("Error " + err.message);
    }
}

async function insertInBatches(data: any[], batchSize = 50) {
    for (let i = 0; i < data.length; i += batchSize) {
        const batch = data.slice(i, i + batchSize);
        const { error } = await supabase.from("design_documents").insert(batch);
        if (error) {
            throw new Error(`Batch insert failed at index ${i}: ${error.message}`);
        }
        console.log(`Inserted batch ${i / batchSize + 1} (${batch.length} rows)`);
    }
}

createAndStoreEmbeddings("C://Users//akhil//OneDrive//Desktop//System Design");