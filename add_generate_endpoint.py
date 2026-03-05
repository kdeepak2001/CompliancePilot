f = open('backend/api/main.py', 'r', encoding='utf-8')
c = f.read()
f.close()

new_endpoint = '''
@app.post("/api/generate-output")
async def generate_output(request: dict):
    """Generate AI agent output using Gemini for demo purposes."""
    try:
        from google import genai
        import os
        
        agent_id = request.get("agent_id", "")
        input_prompt = request.get("input_prompt", "")
        
        agent_prompts = {
            "medical-triage-v1": "You are a medical triage AI agent. Based on the patient symptoms, recommend a care pathway. Be specific and clinical. Keep response under 100 words.",
            "hr-screening-v1": "You are an HR screening AI agent. Based on the candidate profile, give a screening assessment and score out of 100. Keep response under 100 words.",
            "financial-advisory-v1": "You are a financial advisory AI agent. Based on the client financial profile, recommend portfolio actions. Keep response under 100 words."
        }
        
        prompt = agent_prompts.get(agent_id, "Analyze this input and provide a professional recommendation. Keep response under 100 words.")
        
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{prompt}\\n\\nInput: {input_prompt}"
        )
        
        return {"success": True, "output": response.text}
        
    except Exception as e:
        return {"success": False, "output": f"AI agent recommendation based on input: {request.get('input_prompt', '')}. Please consult appropriate specialist for detailed analysis."}
'''

c = c.replace('@app.get("/health")', new_endpoint + '\n@app.get("/health")')

f = open('backend/api/main.py', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')
