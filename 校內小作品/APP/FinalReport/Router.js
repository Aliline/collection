import React from 'react';
import { Image } from 'react-native';
import { Router, Tabs, Stack, Scene ,Actions} from 'react-native-router-flux';
import Home from './Home.js';
import HomeItemDetails from './HomeItemDetails.js';
import LocationList from './LocationList.js';
import LocationDetail from './LocationDetail.js';
// import Search from './Search.js';  // 測試檔
import TourForm from './TourForm.js';
import ScheduleList from './ScheduleList.js';
// import NewSchedule from './NewSchedule.js';  // 測試檔
import Member from './Member.js';  // 測試檔
import blackHome from './Images/blackHome.jpg';
import blackSearch from './Images/blackSearch.jpg';
import blackNew from './Images/blackNew.jpg';
import blackMember from './Images/blackMember.jpg';
import schedule from './schedule.json';
import tlist from './travelList.json';

const iconHome = () => {
  return (
    <Image source={blackHome} style={{ width: 20, height: 20 }} />
  )  
}
const iconSearch = () => {
  return (
    <Image source={blackSearch} style={{ width: 20, height: 20 }} />
  )
}
const iconNew = () => {
  return (
    <Image source={blackNew} style={{ width: 20, height: 20 }} />
  )
}
const iconMember = () => {
  return (
    <Image source={blackMember} style={{ width: 20, height: 20 }} />
  )
}

class Tab extends React.Component {
  constructor(props) {
    super(props);
    this.state= {
      schedules: schedule.schedule,
      tlists: tlist.tList,
    }
  }
  
  // 在搜尋頁面把行程加入
  handleAddtList = (id) => {
    this.setState({
      tlists : [
        ...this.state.tlists,{
          id:id
        }
      ],
    });
    setTimeout(() => {Actions.LocationList()},1000);
    
  }
  handleAddSC = () =>{

  }

  render() {
    const { schedules, tlists } = this.state;
    const { handleAddtList ,handleAddSC} = this;

    return (
      <Router>
        <Tabs headerLayoutPreset="center" tabBarPosition="bottom" showLabel={false}>
          {/* 第一頁 */}
          <Stack key="root" title="首頁" icon={iconHome}>
            <Scene key="Home" component={sceneProps => <Home {...sceneProps} tlists = {tlists}/>} initial hideNavBar={true} />
            <Scene key="HomeItemDetails" title='行程規劃' component={HomeItemDetails} back />
          </Stack>
          <Stack title="搜尋" icon={iconSearch}>
            {/* {alert(tlists[0].id)} */}
            <Scene key="LocationList" component={sceneProps => <LocationList {...sceneProps} tlists = {tlists} handleAddtList = {handleAddtList}/>} hideNavBar />
            <Scene key="LocationDetail" title='景點詳細資料' component={LocationDetail} back />
          </Stack>
          <Stack title="新增行程" icon={iconNew}>
            <Scene
              key="TourForm"
              component={sceneProps => <TourForm {...sceneProps} tlists = {tlists} handleAddSC = {handleAddSC}/>}              
              hideNavBar={true}
            />
            <Scene key="ScheduleList" title="行程清單" component={ScheduleList} />
          </Stack>
          
          <Scene
            key="Member" component={Member} icon={iconMember} renderLeftButton={null} hideNavBar={true}
          />
        </Tabs>
      </Router>
    )
  }
}

export default Tab;