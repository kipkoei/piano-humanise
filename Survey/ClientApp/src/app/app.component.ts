import { CdkStep, StepperSelectionEvent } from '@angular/cdk/stepper';
import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { AbstractControl, FormArray, FormBuilder, FormControl, FormGroup, Validators } from '@angular/forms';
import { SurveyResult, SurveySettings } from './models';

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
    public participantForm = this.fb.group({
        experience: ['', Validators.required],
        listening: ['', Validators.required],
        age: ['', Validators.min(1)],
    });
    public mainForm = this.fb.group({
        samples: this.fb.array([], [Validators.required, Validators.minLength(3)])
    });
    public get samples() {
        return this.mainForm.get('samples') as FormArray;
    }

    public get sampleControls() {
        return this.samples.controls as FormGroup[];
    }
    public sampleFolders(sample: FormGroup) {
        let folders = [];

        for (let control in sample.controls) {
            if (control != 'filename')
                folders.push(control);
        }

        return folders;
    }
    public currentProgress = 0;

    private saving = false;
    private saved = false;

    public participantId?: string;

    constructor(private fb: FormBuilder, private http: HttpClient) { }

    ngOnInit(): void {
        this.http.get<SurveySettings>("api/survey/settings").subscribe(s => {
            this.participantId = s.participantId;
            this.initForm(s.files);
            let _this = this;
            window.addEventListener("beforeunload", function (e) {
                if (_this.saved)
                    return;
                var confirmationMessage = "Are you sure you want to leave? Your results have not been saved yet.";
                e.returnValue = confirmationMessage;
                return confirmationMessage;
            });
        });
    }

    public getSrc(folder: string, sample: FormGroup) {
        let file = sample.controls['filename'].value;

        return `/samples/${folder}/${file}`;
    }

    public save() {
        if (this.saving)
            return;

        this.saving = true;
        let result: SurveyResult = {
            start: this.startTime!,
            end: new Date(),
            answers: [],
            participant: {
                id: this.participantId!,
                age: this.participantForm.controls['age'].value,
                listeningExperience: this.participantForm.controls['listening'].value,
                musicalExperience: this.participantForm.controls['experience'].value,
            }
        };

        for (let sample of this.sampleControls) {
            result.answers.push({
                participantId: this.participantId!,
                filename: sample.controls['filename'].value,
                quantisedScore: sample.controls['quantised'].value,
                humanisedScore: sample.controls['humanised'].value,
                recordingScore: sample.controls['recording'].value                
            });
        }

        this.http.post("api/survey/results", result).subscribe(_ => {
            this.saved = true;
            this.saving = false;
            window.onbeforeunload = null;
        });
    }

    private startTime?: Date;

    public onMove(event: StepperSelectionEvent) {
        //console.log(event.selectedIndex)
        if (event.selectedIndex == 3)
            this.save();
        else if (event.selectedIndex == 2 && !this.startTime)
            this.startTime = new Date();// Date.now()
    }

    public onProgress(event: StepperSelectionEvent) {
        let multiplier = 100 / this.samples.length;

        this.currentProgress = multiplier * event.selectedIndex;
    }

    private initForm(files: string[]) {
        let subfolders = ['quantised', 'humanised', 'recording'];

        for (let file of files) {
            let controls: { [key: string]: FormControl } = {
                'filename': this.fb.control(file, Validators.required)
            };

            for (let folder of shuffle(subfolders))
                controls[folder] = this.fb.control('', [Validators.required, Validators.min(1), Validators.max(5)]);

            this.samples.push(this.fb.group(controls));
        }
    }
}


function shuffle<T>(array: T[]) {
    let currentIndex = array.length, randomIndex;

    // While there remain elements to shuffle.
    while (currentIndex != 0) {

        // Pick a remaining element.
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        // And swap it with the current element.
        [array[currentIndex], array[randomIndex]] = [
            array[randomIndex], array[currentIndex]];
    }

    return array;
}
