import axios from "axios";

export default class Requester {
    url = 'http://localhost:8888';
    filename = null;

    async getAll(file) {
        await this.uploadFile(file);
        const [bpm, tonality, metadata, decade, genre, instruments, deezer] = await Promise.all([
            this.getBPM(),
            this.getTonality(),
            this.getMetadata(),
            this.getDecade(),
            this.getGenre(),
            this.getInstruments(),
            this.deezerSplit(),
        ]);

        return {
            bpm, tonality, metadata, decade, genre, instruments, deezer
        }
    }

    async uploadFile(file) {
        const formdata = new FormData()
        formdata.append('file', file);

        const res = await axios({
            method: "post",
            url: `${this.url}/upload_file`,
            data: formdata
        });
        this.filename = res.data.data.file_path;
    }

    //get the BPM of the song
    async getBPM() {
        const res = await axios({
            method: "get",
            url: `${this.url}/bpm`,
            params: {
                filename: this.filename,
            }
        })

        return res.data.data.bpm;
    }

    //get the tonality of the song
    async getTonality() {
        const res = await axios({
            method: "get",
            url: `${this.url}/tonality`,
            params: {
                filename: this.filename
            }
        })

        return res.data.data.tonality;
    }

    //split the song using Deezer spleeter
    async deezerSplit() {
        const res = await axios({
            method: "get",
            url: `${this.url}/deezer`,
            params: {
                filename: this.filename,
            }
        });
        console.log(res.data);
        Object.keys(res.data.data).forEach(key => res.data.data[key] = this.url + '/' + res.data.data[key]);
        return res.data.data;
    }

    //get the decade of the song
    async getDecade() {
        return axios({
            method: "get",
            url: `${this.url}/decade`,
            params: {
                filename: this.filename,
            }
        }).then((response) => {
            return response.data.data.decade;
        });
    }

    //get the genre of the song
    async getGenre() {
        return axios({
            method: "get",
            url: `${this.url}/genre`,
            params: {
                filename: this.filename,
            }
        }).then((response) => {
            return response.data.data.genre;
        });
    }

    //get the instruments of the song
    async getInstruments() {
        return axios({
            method: "get",
            url: `${this.url}/instruments`,
            params: {
                filename: this.filename,
            }
        }).then((response) => {
            return response.data.data.instruments;
        });
    }

    //get audio metadata
    async getMetadata() {
        const res = await axios({
            method: "get",
            url: `${this.url}/get_metadata`,
            params: {
                filename: this.filename,
            }
        });
        return res.data.data;
    }
}