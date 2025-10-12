# Avatar Voiceflow Bridge

A bridge application that connects Voiceflow chatbots with HeyGen streaming avatars, enabling interactive AI conversations with realistic avatars.

## Features

- **Voiceflow Integration**: Connect to Voiceflow chatbots for conversational AI
- **HeyGen Avatar Streaming**: Display realistic talking avatars powered by HeyGen
- **OpenAI Integration**: Enhanced AI capabilities with OpenAI
- **Real-time Communication**: Live streaming avatar responses
- **Web Interface**: Easy-to-use web interface for interactions

## Prerequisites

- Node.js (v18 or later)
- npm or yarn
- Voiceflow account and API key
- HeyGen account and API key
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd avatar-voiceflow-bridge
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file in the root directory and add your API keys:
```env
VOICEFLOW_API_KEY=your_voiceflow_api_key
VOICEFLOW_PROJECT_ID=your_project_id
HEYGEN_API_KEY=your_heygen_api_key
HEYGEN_AVATAR_ID=your_avatar_id
HEYGEN_VOICE_ID=your_voice_id
OPENAI_API_KEY=your_openai_api_key
PORT=3000
SESSION_TIMEOUT=1800
```

## Usage

1. Start the server:
```bash
npm start
```

2. Open your browser and navigate to `http://localhost:3000`

3. Start interacting with the avatar through the web interface

## Project Structure

```
avatar-voiceflow-bridge/
├── app/
│   └── server.js          # Main server file
├── web/                   # Web interface files
├── docs/                  # Documentation
├── kb/                    # Knowledge base
├── package.json
├── .env                   # Environment variables (create this)
└── README.md
```

## API Endpoints

- `POST /api/chat` - Send message to Voiceflow and get avatar response
- `GET /api/status` - Check server status
- Additional endpoints documented in the API docs

## Development

To run in development mode:
```bash
npm run dev
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the ISC License.

## Support

For support, please open an issue in the GitHub repository or contact the development team.