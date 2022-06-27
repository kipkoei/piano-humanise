namespace Survey;

public record Participant(string Id, int Age, string MusicalExperience, string ListeningExperience);

public record Answer(string ParticipantId, string Filename, int QuantisedScore, int HumanisedScore, int RecordingScore);

public record SurveyResult(DateTime Start, DateTime End ,Participant Participant, IEnumerable<Answer> Answers)
{
    public double? DurationInSeconds { get; set; }
}

public record SurveySettings(string ParticipantId, IEnumerable<string> files);