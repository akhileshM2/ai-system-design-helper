import * as fs from "fs";
import * as path from "path";
import { PDFParse } from "pdf-parse";
import mammoth from "mammoth";

export async function processAllDocs(folderPath: string) {
    const files = fs.readdirSync(folderPath);
    const extractedDocuments: { filename: string; content: string }[] = [];

    for (const file of files) {
        const ext = path.extname(file).toLowerCase();
        if (ext !== ".pdf" && ext !== ".docx") continue;

        const fullPath = path.join(folderPath, file);

        try {
            console.log(`Processing: ${file}...`);
            let content = "";

            if (ext === ".pdf") {
                const dataBuffer = fs.readFileSync(fullPath);
                const parser = new PDFParse({ data: dataBuffer });
                const result = await parser.getText();
                content = result.text;
                console.log(`✅ Extracted text from ${file}`);
            } else if (ext === ".docx") {
                const dataBuffer = fs.readFileSync(fullPath);
                const result = await mammoth.extractRawText({ buffer: dataBuffer });
                content = result.value;
                console.log(`✅ Extracted text from ${file}`);
            }

            extractedDocuments.push({ filename: file, content });
        } catch (error) {
            console.error(`❌ Failed to parse ${file}:`, error);
        }
    }

    return extractedDocuments;
}