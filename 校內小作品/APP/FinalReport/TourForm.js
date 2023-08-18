/* eslint-disable prettier/prettier */
import React from 'react';
import { View, Text, TextInput, Button, StyleSheet } from 'react-native';
import { Actions } from 'react-native-router-flux';
import DatePicker from 'react-native-datepicker';

class TourForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      stitle: null,
      startDate: '2021-01-01',
      endDate: '2021-01-01',
    };
  }

  handleChangeTitle = (text) => {
    this.setState({
      stitle: text,
    });
  };
  handleChangeStartDate = (date) => {
    this.setState({
      startDate: date,
    });
  };
  handleChangeEndDate = (date) => {
    this.setState({
      endDate: date,
    });
  };
  handleRedirectScheduleList = () => {
    const { stitle, startDate, endDate } = this.state;
    const { tlists, handleAddSC } = this.props;
    // 跳轉至餐點詳細頁面時將底部的 Tab 隱藏
    Actions.push('ScheduleList', { hideTabBar: true, stitle: stitle, startDate: startDate, endDate: endDate ,tlists:tlists , handleAddSC : handleAddSC});
  };

  render() {
    const { stitle, startDate, endDate } = this.state;
    const { tlists, handleAddSC } = this.props;
    const { handleRedirectScheduleList } = this;
    
    return (
      <View style={styles.container}>
        <View>
          <View style={styles.item}>
            <Text style={styles.label}>標題 : </Text>
            <TextInput
              keyboardType="default"
              placeholder="請輸入文字"
              value={stitle}
              onChangeText={this.handleChangeTitle}
              style={styles.textInput}
            />
          </View>
          <View style={styles.date}>
            <Text style={styles.dateTitle}>日期 :</Text>
            <View>
              <View style={styles.item}>
                <Text style={styles.label}>開始日期 : </Text>
                <DatePicker
                  style={{ width: 200 }}
                  date={startDate}
                  mode="date"
                  minDate={this.state.date}
                  androidMode={'default'}
                  format="YYYY-MM-DD"
                  minuteInterval={10}
                  onDateChange={this.handleChangeStartDate}
                />
              </View>
              <View style={styles.item}>
                <Text style={styles.label}>結束日期 : </Text>
                <DatePicker
                  style={{ width: 200 }}
                  date={endDate}
                  mode="date"
                  minDate={startDate}
                  androidMode={'default'}
                  format="YYYY-MM-DD"
                  minuteInterval={10}
                  onDateChange={this.handleChangeEndDate}
                />
              </View>
            </View>
          </View>
        </View>
        <Button title="下一步" disabled={!stitle || !startDate || !endDate} onPress={handleRedirectScheduleList} />
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'space-between',
    backgroundColor: '#FFF',
    borderWidth: 1,
    borderColor: '#EEE',
  },
  item: {
    height: 40,
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 5,
    paddingHorizontal: 10,
  },
  label: {
    fontWeight: 'bold',
  },
  picker: {
    width: 150,
    marginLeft: 10,
  },
  textInput: {
    flex: 1,
    borderBottomWidth: 1,
    borderBottomColor: '#C0C0C0',
    marginLeft: 15,
  },
  switch: {
    marginLeft: 10,
  },
  date: {
    flexDirection: 'row',
    marginVertical: 5,
    paddingHorizontal: 10,
  },
  dateTitle: {
    fontWeight: 'bold',
    paddingTop: 16,
  },
});

export default TourForm;
