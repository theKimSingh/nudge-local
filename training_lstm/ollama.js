import 'dotenv/config';
import fs from 'fs';

// 1. Read and parse your synthetic test suite safely
const testSuite = JSON.parse(fs.readFileSync('./synthetic_event_extraction_dataset.json', 'utf-8'));

// 2. Core extraction logic (Updated to use native Ollama API via fetch)
async function extractEventData(text, testDate) {
  const today = testDate || new Date().toISOString().split('T')[0];
  // Hardcoded timezone to match the mock environment evaluation requirements
  const timezone = "America/New_York"; 

  // Using the exact system prompt from your Express app
  const systemPrompt = `You are a helpful assistant that extracts event details. 
    Today's date is ${today} and the user's timezone is ${timezone}. 
    Extract event details from the user's text. Keep the end of the event at 1 hour after beginning unless otherwise specified.
    You MUST respond with a list of JSON objects containing ONLY these keys: 
    "id", "summary", "begin" (ISO 8601 format), "duration" (integer minutes), "repeats" ("daily", "weekdays", "weekly", or null)
    If a field is missing or unknown, use null. Output your response strictly as a raw JSON object, there may be multiple events.
    Do NOT wrap it in markdown code blocks (\`\`\`json), and do not output any other text.`;

  const response = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'qwen3:0.6b', // Matches your Express route model choice
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: text }
      ],
      format: 'json', // Tells Ollama to strictly output valid JSON
      stream: false,  
      options: {
        temperature: 0.1,
        num_predict: 250
      }
    })
  });

  if (!response.ok) {
    throw new Error(`Ollama API error: ${response.statusText}`);
  }

  const data = await response.json();
  const content = data.message?.content;

  if (!content || content.trim() === "") {
    throw new Error('Ollama model returned no content');
  }

  const parsedData = JSON.parse(content);
  
  // Since the prompt asks for an array/list, let's normalize it to an object 
  // for the singular test runner, matching how your code tests single properties.
  const rawEvent = Array.isArray(parsedData) ? parsedData[0] : parsedData;
  
  // Return the expected schema directly. Notice we don't divide by 60 here 
  // because the prompt already tells Ollama to provide "duration" in integer minutes!
  return {
    id: rawEvent.id || null,
    summary: rawEvent.summary,
    begin: rawEvent.begin ? rawEvent.begin.split('.')[0] : null, 
    duration: rawEvent.duration ? Math.round(rawEvent.duration) : null,
    repeats: rawEvent.repeats || null
  };
}

// 3. Automated Test Execution Loop
async function runTestSuite() {
  console.log(`🚀 Loaded ${testSuite.length} cases from synthetic.json. Running accuracy tests against Ollama...\n`);
  let passedCount = 0;
  
  // Hardcoded date anchor to safely parse terms like 'tomorrow' against the synthetic data
  const mockToday = "2026-05-20"; 

  for (const test of testSuite) {
    console.log(`Testing: "${test.name}"`);
    console.log(`Input:   "${test.input}"`);
    
    try {
      const actual = await extractEventData(test.input, mockToday);

      // Read expected properties 
      const expectedSummary = test.output?.summary || "";
      const expectedBegin = test.output?.begin || "";
      const expectedDuration = test.output?.duration; // e.g., 30 (Minutes)
      
      // Perform safe verification checks
      const summaryPass = actual && actual.summary 
        ? actual.summary.toLowerCase().includes(expectedSummary.toLowerCase())
        : false;
        
      const beginPass = actual && actual.begin 
        ? actual.begin.startsWith(expectedBegin)
        : false;
        
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