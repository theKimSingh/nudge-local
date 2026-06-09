import { GoogleGenerativeAI } from "@google/generative-ai";
import 'dotenv/config';
import fs from 'fs';

// 1. Read and parse your synthetic test suite safely
const testSuite = JSON.parse(fs.readFileSync('./synthetic_event_extraction_dataset.json', 'utf-8'));

// 2. Core extraction logic (with your built-in output adapter)
async function extractEventData(text, testDate) {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({ model: 'gemma-4-31b-it' });
  
  const today = testDate || new Date().toISOString().split('T')[0];
  const timezone = "America/New_York"; 

  const prompt = `Today's date is ${today} and the user's timezone is ${timezone}. 
  Extract event details from the following text: '${text}'. Keep the end of the event at 1 hour after beginning unless otherwise specified.
  Return the result in a strict JSON format with the following keys: 
  'summary', 'begin' (ISO 8601 format), 'end' (ISO 8601 format), 'description', 'location'. 
  If a field is missing, use null. Only return the JSON.`;

  const result = await model.generateContent({
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: { responseMimeType: "application/json", responseSchema: {
            type: "OBJECT",
            properties: {
                output: {
                    type: "OBJECT",
                    properties: {
                        // Allows the model to return a string ID or null
                        id: { type: "STRING", nullable: true }, 
                        summary: { type: "STRING" },
                        // ISO 8601 timestamp string
                        begin: { type: "STRING" }, 
                        duration: { type: "INTEGER" },
                        repeats: { type: "STRING", nullable: true }
                    },
                    // Ensures the model always populates these fields
                    required: ["summary", "begin", "duration"] 
                }
            },
            required: ["output"]
        } }
  });

  let content = result.response.text();
  content = content.replace(/^```json\s*/i, '')  
                 .replace(/^```\s*/, '')      
                 .replace(/```\s*$/, '');
  if (!content || content.trim() === "") {
    throw new Error('Gemini API returned no content');
  }

  // console.log(`Raw Gemini Output:`, content);

  const parsed = JSON.parse(content);
  const rawJson = parsed.output || parsed;
  console.log(`Parsed JSON:`, rawJson);
  const durationMinutes = rawJson.duration ? Math.round(rawJson.duration / 60) : null;
  return {
    id: null,
    summary: rawJson.summary,
    begin: rawJson.begin ? rawJson.begin.split('.')[0] : null, 
    duration: durationMinutes,
    repeats: null
  };
}

// 3. Automated Test Execution Loop
async function runTestSuite() {
  console.log(`🚀 Loaded ${testSuite.length} cases from synthetic.json. Running accuracy tests...\n`);
  let passedCount = 0;
  
  // Hardcoded date anchor to safely parse terms like 'tomorrow' against the synthetic data
  const mockToday = "2026-05-20"; 

  for (const test of testSuite) {
    console.log(`Testing: "${test.name}"`);
    console.log(`Input:   "${test.input}"`);
    
    try {
      const actual = await extractEventData(test.input, mockToday);

      // 2. Safely read expected properties 
      const expectedSummary = test.output?.summary || "";
      const expectedBegin = test.output?.begin || "";
      const expectedDuration = test.output?.duration; // e.g., 30 (Minutes)
      
      // 3. Perform safe verification checks
      const summaryPass = actual && actual.summary 
        ? actual.summary.toLowerCase().includes(expectedSummary.toLowerCase())
        : false;
        
      const beginPass = actual && actual.begin 
        ? actual.begin.startsWith(expectedBegin)
        : false;
        
      // Ensure actual.duration isn't still accidentally evaluating to seconds
      const durationPass = actual && actual.duration === expectedDuration;
      
      if (summaryPass && beginPass && durationPass) {
        console.log(`✅ PASSED\n`);
        passedCount++;
      } else {
        const expectedDisplay = {
          summary: expectedSummary,
          begin: expectedBegin,
          duration: expectedDuration
        };
        console.log(`❌ FAILED`);
        console.log(`   Expected:`, JSON.stringify(expectedDisplay));
        console.log(`   Actual:  `, JSON.stringify(actual));
        console.log();
      }
    } catch (error) {
      console.log(`💥 ERROR: ${error.message}\n`);
    }
  }

  console.log(`--- Test Summary ---`);
  console.log(`Passed: ${passedCount} / ${testSuite.length}`);
}

runTestSuite();