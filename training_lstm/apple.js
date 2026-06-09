import 'dotenv/config';
import fs from 'fs';

// 1. Read and parse your synthetic test suite safely
const testSuite = JSON.parse(fs.readFileSync('./synthetic_event_extraction_dataset.json', 'utf-8'));

// 2. Core extraction logic making HTTP calls directly to local Apple Intelligence
async function extractEventData(text, testDate) {
  const today = testDate || new Date().toISOString().split('T')[0];
  const timezone = "America/New_York"; 

  // --- WHERE THE PROMPT LIVES ---
  // This tells the local Apple Intelligence model exactly how to parse your unstructured text
//   const systemPrompt = `You are a helpful assistant that extracts event details. 
//     Today's date is ${today} and the user's timezone is ${timezone}. 
//     Extract event details from the user's text. Keep the end of the event at 1 hour after beginning unless otherwise specified.
//     You MUST respond with a list containing a single JSON object containing ONLY these keys: 
//     "id", "summary", "begin" (ISO 8601 format), "duration" (integer minutes), "repeats" ("daily", "weekdays", "weekly", or null)
//     If a field is missing or unknown, use null. Output your response strictly as a raw JSON object, do not wrap it in markdown block tags.`;
    const systemPrompt = `You are a helpful assistant that extracts event details. 
    Today's date is ${today} and the user's timezone is ${timezone}. 
    Extract event details from the user's text. To determine the correct "begin" date, use ${today} as your anchor point:
    - "Today", "tomorrow", and "day after tomorrow" must be translated to their exact calendar dates relative to ${today}.
    - Days of the week (e.g., "Monday", "next Friday") must be resolved to the correct upcoming calendar date relative to ${today}. If the day mentioned has already passed this week, assume it refers to the next occurrence.
    Keep the end of the event at 1 hour after beginning unless otherwise specified.
    You MUST respond with a list containing a single JSON object containing ONLY these keys: 
    "summary", "date" (Strict format: YYYY-MM-DD), "time" (Strict format: HH:mm (24-hour clock), "duration" (integer minutes), "repeats" ("daily", "weekdays", "weekly", or null)
    If a field is missing or unknown, use null. Output your response strictly as a raw JSON object, do not wrap it in markdown block tags.`;
  // HTTP POST Request directly targeting our local Apple Silicon gateway
  const response = await fetch('http://localhost:9999/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'foundation', // Points to the native Apple On-Device Foundation Model
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: text }
      ],
      temperature: 0.1,
      stream: false
    })
  });

  if (!response.ok) {
    throw new Error(`Apple Intelligence local server error: ${response.statusText}`);
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content;

  if (!content || content.trim() === "") {
    throw new Error('Apple model returned an empty string output');
  }

  // Clean raw string in case the model ignored directions and snuck in backticks
  const cleanContent = content.replace(/```json|```/g, '').trim();
  const parsedData = JSON.parse(cleanContent);
  const rawEvent = Array.isArray(parsedData) ? parsedData[0] : parsedData;
  
  return {
    summary: rawEvent.summary || text,
    begin: rawEvent.begin ? rawEvent.begin.split('.')[0] : null, 
    date: rawEvent.date || null,
    time: rawEvent.time || null,
    duration: rawEvent.duration ? Math.round(rawEvent.duration) : 60,
    repeats: rawEvent.repeats || null
  };
}

// 3. Automated Test Execution Loop
async function runTestSuite() {
  console.log(`🚀 Loaded ${testSuite.length} cases from synthetic.json. Running accuracy tests against Apple Intelligence HTTP Gateway...\n`);
  let passedCount = 0;
  const mockToday = "2026-05-20"; 

  for (const test of testSuite) {
    console.log(`Testing: "${test.name}"`);
    console.log(`Input:   "${test.input}"`);
    
    try {
      const actual = await extractEventData(test.input, mockToday);

      const expectedSummary = test.output?.summary || "";
      const expectedBegin = test.output?.begin || "";
      const expectedDuration = test.output?.duration; 
      
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
        const expectedDisplay = { summary: expectedSummary, begin: expectedBegin, duration: expectedDuration };
        console.log(`❌ FAILED`);
        console.log(`   Expected:`, JSON.stringify(expectedDisplay));
        console.log(`   Actual:  `, JSON.stringify(actual));
        console.log();
      }
    } catch (error) {
      console.log(`💥 ERROR parsing or communicating with Apple model: ${error.message}\n`);
    }
  }

  console.log(`--- Test Summary ---`);
  console.log(`Passed: ${passedCount} / ${testSuite.length}`);
}

runTestSuite();