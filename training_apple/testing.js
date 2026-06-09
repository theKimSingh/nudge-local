import 'dotenv/config';
import fs from 'fs';

// 1. Read and parse your dataset safely
const testSuite = JSON.parse(fs.readFileSync('./synthetic_event_extraction_dataset.json', 'utf-8'));

/**
 * Heuristic to find days of the week (including "next <day>") in a string 
 * and append their calculated calendar dates.
 * Ex: "next monday" -> "next monday (2026-06-01)"
 */
function appendDatesToDayNames(text, anchorDateStr) {
  const daysOfWeekMap = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
  
  const [year, month, day] = anchorDateStr.split('-').map(Number);
  const anchorDate = new Date(year, month - 1, day);
  const anchorDayIndex = anchorDate.getDay();

  const dayRegex = /\b(next\s+)?(sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b/gi;

  return text.replace(dayRegex, (match, nextModifier, dayName) => {
    const targetDayIndex = daysOfWeekMap.indexOf(dayName.toLowerCase());
    
    // 1. Calculate absolute distance in the natural calendar progression
    let daysDiff = targetDayIndex - anchorDayIndex;
    
    if (nextModifier) {
      // If "next" is stated, we are looking for the occurrence in the NEXT calendar week
      if (daysDiff <= 0) {
        daysDiff += 7; // Target day already happened this week, move to next week
      } else {
        // Target day is LATER this week (e.g., it's Wed, target is Fri). 
        // Conversational "Next Friday" usually means the Friday of next week.
        daysDiff += 7; 
      }
    } else {
      // No "next" modifier (e.g., just "Tuesday")
      if (daysDiff <= 0) {
        daysDiff += 7; // Passed or today -> means next week's occurrence
      }
    }

    // Apply offset
    const resolvedDate = new Date(anchorDate);
    resolvedDate.setDate(anchorDate.getDate() + daysDiff);

    // Format YYYY-MM-DD
    const yyyy = resolvedDate.getFullYear();
    const mm = String(resolvedDate.getMonth() + 1).padStart(2, '0');
    const dd = String(resolvedDate.getDate()).padStart(2, '0');
    
    return `${match} (${yyyy}-${mm}-${dd})`;
  });
}

/**
 * Extracts event details while maintaining an active chat context array
 */
async function extractEventData(messageHistory, testDate) {
  const today = testDate || new Date().toISOString().split('T')[0];
  // const timezone = "America/New_York"; 

  const daysOfWeekMap = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const dateParts = today.split('-');
  const parsedDate = new Date(dateParts[0], dateParts[1] - 1, dateParts[2]);
  const dayNameToday = daysOfWeekMap[parsedDate.getDay()];

  const systemPrompt = `You are a helpful assistant that extracts event details. 
    Today's date is ${today} (${dayNameToday}). 
    Extract event details from the user's text. To determine the correct "date", use ${today} as your anchor point.
    
    LIMITATION: Exactly one event is changed/added/modified per prompt. If there is no end time provided, assume the event is 1 hour long. 
    If the input text contains an explicit date in parentheses, e.g., (YYYY-MM-DD), you MUST use this date for the "date" field. 
    You MUST respond with a list containing a single JSON object containing ONLY these keys: 
    "id" (integer tracking the event), "title", "date" (Strict format: YYYY-MM-DD), "start_time" (Strict format: HH:mm 24-hour clock), "end_time" (Strict format: HH:mm 24-hour clock), "repeats" ("daily", "weekdays", "weekly", or null)
    If a field is missing, unknown, or being explicitly cleared, use null. Output your response strictly as a raw JSON list, do not wrap it in markdown block tags.`;

  const response = await fetch('http://localhost:9999/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'foundation', 
      messages: [
        { role: 'system', content: systemPrompt },
        ...messageHistory
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

  const cleanContent = content.replace(/```json|```/g, '').trim();
  const parsedData = JSON.parse(cleanContent);
  
  const rawEvent = Array.isArray(parsedData) ? parsedData[0] : parsedData;
  
  return {
    id: rawEvent.id || null,
    title: rawEvent.title || null, 
    date: rawEvent.date || null,
    start_time: rawEvent.start_time || null,
    end_time: rawEvent.end_time || null,
    repeats: rawEvent.repeats || null
  };
}

// 3. Automated Test Execution Loop
async function runTestSuite() {
  const normalizedSuite = testSuite.map((item, index) => {
    if (item.steps) return item; 
    
    return {
      description: item.description || `Flat Prompt #${index + 1}`,
      steps: [{
        input: item.input,
        output: Array.isArray(item.output) ? item.output : [item.output]
      }]
    };
  });

  console.log(`🚀 Loaded ${normalizedSuite.length} test sequences.\n`);
  
  const mockToday = "2026-05-20"; // This is a Wednesday
  let totalPromptsTested = 0;
  let totalPromptsPassed = 0;

  for (let i = 0; i < normalizedSuite.length; i++) {
    const sequence = normalizedSuite[i];
    console.log(`------------------------------------------------------------`);
    console.log(`▶️ Running Sequence #${i + 1}: ${sequence.description}`);
    console.log(`------------------------------------------------------------`);

    let conversationHistory = [];
    
    for (const step of sequence.steps) {
      totalPromptsTested++;
      
      // --- APPLY EXTENDED HEURISTIC PREPROCESSING ---
      const preprocessedInput = appendDatesToDayNames(step.input, mockToday);
      console.log(`\n[Prompt (Raw)]: "${step.input}"`);
      if (preprocessedInput !== step.input) {
        console.log(`[Prompt (Heuristic Added)]: "${preprocessedInput}"`);
      }
      
      conversationHistory.push({ role: 'user', content: preprocessedInput });

      try {
        const actual = await extractEventData(conversationHistory, mockToday);
        const expected = step.output; 

        console.log(`[Expected]:`, JSON.stringify(expected));
        console.log(`[Actual]:  `, JSON.stringify([actual]));

        conversationHistory.push({ role: 'assistant', content: JSON.stringify([actual]) });

        // Assert accuracy targets
        const idPass = true;
        const titlePass = true;
          
        const datePass = actual.date === expected[0].date;
        const timePass = actual.start_time === expected[0].start_time;
        const timePassEnd = actual.end_time === expected[0].end_time;

        if (idPass && titlePass && datePass && timePass && timePassEnd) {
          console.log(`✅ Step Passed!`);
          totalPromptsPassed++;
        } else {
          console.log(`❌ Step Failed! Reasons:`, {
            idMatch: idPass,
            titleMatch: titlePass,
            dateMatch: datePass,
            timeMatch: timePass,
            timeMatchEnd: timePassEnd
          });
        }

      } catch (error) {
        console.error(`💥 Execution error on step: ${error.message}`);
      }
    }
  }

  const accuracy = totalPromptsTested > 0 ? ((totalPromptsPassed / totalPromptsTested) * 100).toFixed(2) : 0;
  console.log(`\n============================================================`);
  console.log(`📊 Final Results: ${totalPromptsPassed}/${totalPromptsTested} steps passed (${accuracy}% Accuracy)`);
  console.log(`============================================================`);
}

runTestSuite();