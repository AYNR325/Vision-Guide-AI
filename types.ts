
export interface AudioConfig {
  sampleRate: number;
  channels: number;
}

export enum ConnectionStatus {
  DISCONNECTED = 'DISCONNECTED',
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  ERROR = 'ERROR'
}

export interface TranscriptionItem {
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}
