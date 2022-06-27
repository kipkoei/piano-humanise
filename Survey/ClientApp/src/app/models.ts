
export class Participant {
    id!: string;
    age!: number;
    musicalExperience!: string;
    listeningExperience!: string;
}

export class Answer {
    participantId!: string;
    filename!: string;
    quantisedScore!: number;
    humanisedScore!: number;
    recordingScore!: number;
}

export class SurveyResult {
    start!: Date;
    end!: Date;
    participant!: Participant;
    answers!: Answer[];
}

export class SurveySettings {
    participantId!: string;
    files!: string[];
}
