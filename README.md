# Ducky
Initial Idea/ Planning

## New user session:
A user types in the CLI a command (ducky –thoughts), which makes ducky create a sessionID and sessionName and call a speech to text service (looking for a free option for now just for a proof of concept) to record and process the user's audio in real-time. Once the user stops speaking for a set amount of time, or if the user says an end phrase (like “have any thoughts ducky?”), then the audio processing is stopped and the transcript is sent to an LLM which will generate a response. Here are the response guidelines:

- The response will NOT give the direct answer unless the user specifically toggles it on in a configuration file (the default is false or off)
- The response will guide the user with hints using a mixture of:
  - Socratic: asks questions that lead the user to find the flaw
  - Hint ladder: escalating hints, from a nudge to a full solution on request (will only give the full solution if toggled)
- The type of response given will be determined by the LLM and the transcript of what the user asked 
- I may include Laya or Jev (a classifier) here to determine the better type of response style
- Ducky will not read the user’s code, and only consider their vocal thought process (I may implement coding reading or text inputs at a later time, but for now it is strictly helping voiced thoughts)
- If ducky wants to provide hints with code as examples, it will use example variables and functions to give the user an idea of how to implement it

If the user voices their variables, functions, classes, etc, only then will ducky directly call on those variables. However, if there is not enough context to determine what they do, then ducky will first clarify that they are assuming that the user is trying to do something, and then probe the user to better explain, to hopefully nudge them in the right direction
After a response has been made, the LLM will send it back to ducky to be displayed in the terminal to the user. At the end of text, ducky will display a list of choices the user can make. The user can then decide to do 1 of 3 things:

- End the session by closing the terminal or typing ducky –end. Ducky stops running, and the user can do whatever they want. 
- Start a new chat with a different question and context by running ducky –thoughts
- Continue the chat by typing ducky –continue
- Boot up an existing session and continuing it by running ducky –session sessionName

## Continued conversation (current session):
If a user wants to continue a session to ducky, then the user can type ducky –continue. Ducky will perform the same task as described in a new user session, with the addition of saving a summary of the session. After the session goes on for a while, Ducky will prune older transcriptions to save space. Instead a summary of the sessions up until that point will be saved. Each time ducky prunes old transcripts, the summary will be updated to ensure that it has relevant and up to date information. Here are some guidelines for ducky in continue mode:

- A transcript of the most recent ducky –thoughts will be saved in an SQLite database
- A summary of the whole session will be stored in an SQLite database
- No audio will be saved, only text transcripts to save space

## Continued conversation (existing old session):
If a user wants to boot up an old session, then they can run ducky –session sessionName. Similar logic will apply as the continued conversation, except that Ducky will be using an older session instead of a current session and will need to change the context to the sessionName and sessionID of the new session being run. Before ducky switches session, ducky will save a summary of the current session it is in, before it switches to the new session. 

## Configuration:
I am planning on users to be able to configure these things for now:
- sessionName: users can change the session name to a new name with ducky –name-change sessionName newName
- Direct answers: users can toggle ducky to give a direct answer or solution to their problem with ducky –answers true
- LLM API keys: users can add their keys to their own .envs (for now let’s only implement claude, chatgpt, grok, qwen, and gemini)
- LLMs: users can change which LLM is used with ducky –agent agentName
- Speech to text service: I want to use a free option for now, that preferably stores things locally rather than on the providers server
- Ducky’s response style: in the future i plan on implementing different text/ response styles. For example, we can make ducky respond like a disappointed parent or angry teacher in order to make users feel the need to work harder to get to their solution. This is mostly for fun, and is not a top priority for now. I want to use ducky –tone -toneType

## Commands:
Here are some commands I have thought up that have not been mentioned already:

- ducky –history: shows the history of the current session (only if it’s continuous, it will not show the history of chats pulled from an old pre-existing session. In that case, it will only show the summary)
- ducky –sessions: lists all the sessions the user has
- ducky –sessions -delete sessionName: deletes a session
- ducky –help: lists all the commands
- duck –quack: easter egg for the user. Ducky will literally quack (I will provide a sound file for this)
