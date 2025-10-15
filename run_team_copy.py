from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from tool import  Get_keyword_tool, Get_mapping_tool, Opendistro_search, agg_json_to_excel
from function import classify_brand, filter_field_description, flatten_es_mapping, extract_json_string, stream_data, write_export_script
from RAG import query_dsl_examples
import re, io, os, json, asyncio, aiofiles, time
from pathlib import Path
import pandas as pd
from config import team_state_dir, mapping_dir, rag_dir, get_model_client, result_dir

from ES_Query import generate_elasticsearch_query_from_natural_language
from HistoryMatchTeam import get_HistoryMatchTeam
from ReqTeam import get_Reqteam
from FieldTeam import get_fieldTeam
from DSLTeam import get_DSLteam
from ExecuteTeam import get_Exeteam
from ReportSaverTeam import get_ReportSaverTeam
from mode_selector import get_ModelSelector

import streamlit as st

client = get_model_client()

async def main():
    try:    
        # st.title("Report Generator")
        
        target_dsl_path = Path(os.path.join(result_dir,"DSLQuery.json"))
        target_py_script = (os.path.join(result_dir, "result.py"))

        #delete Reqstate in last session
        if "delete_Reqstate" not in st.session_state:
            st.session_state.delete_Reqstate = False
        if not st.session_state.delete_Reqstate:
            Req_state_path = os.path.join(team_state_dir, "Req_state.json")
            if os.path.exists(Req_state_path):
                os.remove(Req_state_path)
            st.session_state.delete_Reqstate = True

        if "history_check" not in st.session_state:
            st.session_state.history_check = True  
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
 
        if "finish" not in st.session_state:
            st.session_state.finish = False
 
        if "brand" not in st.session_state:
            st.session_state.brand = ""
 
        if "field_select" not in st.session_state:
            st.session_state.field_select = True
 
        if "Req_stage" not in st.session_state:
            st.session_state.Req_stage = "customer_finder_agent"
 
        if "passing_turn" not in st.session_state:
            st.session_state.passing_turn = False
 
        if "Req_passed_message" not in st.session_state:
            st.session_state.Req_passed_message = ""
        
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""
 
        if "customer_name" not in st.session_state:
            st.session_state.customer_name = ""
        
        if "mode" not in st.session_state:
            st.session_state.mode = "auto"
        # frontend
        OPTIONS = [
            {"key": "auto", "title": "Auto"},
            {"key": "fast", "title": "Fast"},
            {"key": "thinking", "title": "Thinking"},
        ]

        def get_title(k): return next(o["title"] for o in OPTIONS if o["key"] == k)

        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            with st.popover(f"Report Generator · {get_title(st.session_state.mode)}", use_container_width=True):
                st.radio(
                    "Select Mode",
                    options=[o["key"] for o in OPTIONS],
                    index=[o["key"] for o in OPTIONS].index(st.session_state.mode),
                    format_func=get_title,
                    key="mode",
                )
                st.checkbox("History check", key="history_check")

        # show chat history
        if st.session_state.messages: 
            with st.popover("Chat History"):
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
 
        prompt = st.chat_input("Please enter your requirement:",key="prompt")
        # if (prompt or st.session_state.passing_turn) and not st.session_state.finish:
            # st.chat_input("Please enter your requirement:", disabled=True)
        if (prompt or st.session_state.passing_turn):
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
 
            st.session_state.field_select = True        #field_select always True as default
            # state_path= f"{st.session_state.Req_stage}.json"
            state_path= os.path.join(team_state_dir,"Req_state.json")
            Req_team = await get_Reqteam(stage = st.session_state.Req_stage, state_path= state_path)
            if st.session_state.passing_turn:
                Req_stream = Req_team.run_stream(task= st.session_state.Req_passed_message)
            else:
                Req_stream = Req_team.run_stream(task= prompt)
            
            
            NEXT_STAGE = False
            st.session_state.passing_turn = False
            async for Reqteam_msg in Req_stream:
                         
                if not isinstance(Reqteam_msg, TextMessage):
                    continue
                # st.write(Reqteam_msg.source)
                
                if "CUSTOMER_FOUND" in Reqteam_msg.content:
                    st.session_state.Req_stage = "filter_finder_agent"
                    st.session_state.passing_turn = True
                elif "PASS to analyzer" in Reqteam_msg.content:
                    st.session_state.Req_stage = "requirements_analyzer"
                    st.session_state.passing_turn = True
 
                if Reqteam_msg.source == "RequirementsFinalizer":
                    if "mentioned_methodology" in Reqteam_msg.content:
                        NEXT_STAGE = True         
                    if ("_CONFIRMED:" in Reqteam_msg.content) and ("Question:" not in Reqteam_msg.content):
                        st.session_state.passing_turn = True
                        st.rerun()
                
                if Reqteam_msg.source == "user":
                    st.session_state.user_input = Reqteam_msg.content + ";" + st.session_state.user_input
                    continue
 
                st.session_state.messages.append({"role": Reqteam_msg.source, "content": Reqteam_msg.content})
                
                with st.chat_message("ai"):
                    st.write_stream(stream_data(Reqteam_msg.content))
                    
            
            async with aiofiles.open(state_path, "w") as file:
                Req_state = await Req_team.save_state()
                await file.write(json.dumps(Req_state, default=str, indent=2,))

            Req_lastmsg = Req_state["llm_context"]["messages"][-1]["content"]
                            
            if st.session_state.Req_stage == "filter_finder_agent" and st.session_state.passing_turn:
                st.session_state.customer_name = extract_json_string(Req_lastmsg)
                st.session_state.Req_passed_message = f"{st.session_state.user_input}; customer_name:{st.session_state.customer_name}"
                st.rerun()
            elif st.session_state.Req_stage == "requirements_analyzer" and st.session_state.passing_turn:
                confirmed_filter_dict = extract_json_string(Req_lastmsg)
                confirmed_filter_str= json.dumps(confirmed_filter_dict, default=str, indent=2)
                st.session_state.Req_passed_message = f"{st.session_state.Req_passed_message}; confirmed: {confirmed_filter_str}"
                st.rerun()
                
            
            # requiremnet stage completed
            if NEXT_STAGE:
                # retriving data from req_state.json
                user_requirement = extract_json_string(Req_lastmsg)
                
                
                requirement = json.dumps(user_requirement, ensure_ascii=False, indent=2)
                with open(os.path.join(mapping_dir,"flattened_mapping.json"), "r", encoding="utf-8") as f:
                    data = json.load(f)
                mapping = json.dumps(data, ensure_ascii=False, indent=2)
                summary = user_requirement["summary"]
                
                # pre-processing base on customer              
                brand = classify_brand(st.session_state.customer_name)
                print(f"Brand: {brand}")
                print(f"customer_name: {st.session_state.customer_name}")
                print(f"customer_name type: {type(st.session_state.customer_name)}")

                st.session_state.brand = brand
                filter_field_description(brand)
                # embed_fields_from_csv()

                #History match
                if st.session_state.history_check: 
                    with st.status("Finding history...", expanded=False) as history_status:
                        try:
                            rag_result = query_dsl_examples(query = summary, brand=brand, top_k = 3)
                            with st.chat_message("ai"):
                                st.write(summary)
                            
                            
                            history_team = await get_HistoryMatchTeam(rag_result = rag_result)
                            history_stream = history_team.run_stream(task = requirement)
                            

                            async for his_msg in history_stream:
                                if not isinstance(his_msg, TextMessage):
                                    continue
                                if his_msg.source == "user":
                                    continue
                                st.session_state.messages.append({"role": his_msg.source, "content": his_msg.content})
                                with st.chat_message("ai"):
                                    st.write_stream(stream_data(his_msg.content))
                            HistoryTeam_state = await history_team.save_state()
                            with open(os.path.join(team_state_dir, "FieldTeam_state.json"), "w", encoding="utf-8") as f:
                                json.dump(HistoryTeam_state, f, ensure_ascii=False, indent=2, default=str)
                            
                            HistoryTeam_msgs = HistoryTeam_state["agent_states"]["RoundRobinGroupChatManager"]["message_thread"]
                            HistoryTeam_lastmsgs = HistoryTeam_msgs[-1]["content"]
                            
                            if "No such historical report" not in HistoryTeam_lastmsgs:
                                st.session_state.field_select = False
                                history_status.update(label="History founded.", state="complete")
                            else:
                                history_status.update(label="History not founded.", state="complete")
                        except Exception as e:
                            history_status.update(label=f"Error:{e}", state="error")

                # auto mode
                if st.session_state.field_select and st.session_state.mode == "auto":
                    try:
                        model_selector = await get_ModelSelector()
                        model_selector_stream = model_selector.run_stream(task = requirement)
                        async for model_selector_msg in model_selector_stream:
                            if not isinstance(model_selector_msg, TextMessage):
                                continue
                            if "Thinking mode" in model_selector_msg.content:
                                mode = "thinking"
                                st.write_stream(stream_data("Oh, difficult task. Thinking..."))
                            if "Fast mode" in model_selector_msg.content:
                                mode = "fast"  
                                st.write_stream(stream_data("Ha, easy task~"))                              
                    except Exception as e:
                        st.write(f"error: {e}")
                                   
                if st.session_state.field_select and (st.session_state.mode == "thinking" or mode == "thinking"):
                        # Field Selection
                    with st.status("Selecting field...", expanded=False) as field_status:
                        try:
                            Field_team = await get_fieldTeam(task = summary, mapping = mapping, user_require = user_requirement)
                            Field_stream = Field_team.run_stream(task=summary)
                            async for field_msg in Field_stream:
                                if not isinstance(field_msg, TextMessage):
                                    continue
                                if field_msg.source == "user":
                                    continue
                                st.session_state.messages.append({"role": field_msg.source, "content": field_msg.content})
                                with st.chat_message("ai"):
                                    st.write_stream(stream_data(field_msg.content))
                            FieldTeam_state = await Field_team.save_state()
                            with open(os.path.join(team_state_dir, "FieldTeam_state.json"), "w", encoding="utf-8") as f:
                                json.dump(FieldTeam_state, f, ensure_ascii=False, indent=2, default=str)
                            
                            FieldTeam_msgs = FieldTeam_state["agent_states"]["RoundRobinGroupChatManager"]["message_thread"]
                            field_list = FieldTeam_msgs[-1]["content"]
                            confirmed_fieldList_dict = extract_json_string(field_list)
                            confirmed_fieldList_str= json.dumps(confirmed_fieldList_dict, default=str, indent=2)
                            field_status.update(label="Field selection completed.", state="complete")
                        except Exception as e:
                            field_status.update(label=f"Field selection failed:{e}", state="error")
                    
                    with st.status("Writing DSL...", expanded=False) as writing_status:
                        try:        
                            DSLteam = await get_DSLteam(task = requirement, customer_name = st.session_state.customer_name, mapping=mapping,field_list=confirmed_fieldList_str)
                            DSLstream = DSLteam.run_stream(task = requirement) 
                                           
                            print(f"DSLstream: {DSLstream}")
                            async for DSL_msg in DSLstream:
                                print(f"DSL_msg: {DSL_msg}")
                                if not isinstance(DSL_msg, TextMessage):
                                    continue
                                if DSL_msg.source == "user":                                   
                                    continue
                                st.session_state.messages.append({"role": DSL_msg.source, "content": DSL_msg.content})
                                with st.chat_message("ai"):
                                    st.write_stream(stream_data(DSL_msg.content))
                            DSLteamstate = await DSLteam.save_state()
                            with open(os.path.join(team_state_dir, "DSLTeam_state.json"), "w", encoding="utf-8") as f:
                                json.dump(DSLteamstate, f, ensure_ascii=False, indent=2, default=str)
                                
                            print("################################################################################################")

                            DSLteam_msgs = DSLteamstate["agent_states"]["RoundRobinGroupChatManager"]["message_thread"]
                            DSLteam_lastmsgs = DSLteam_msgs[-1]["content"]
                            #print(f"DSLteam_lastmsgs: {DSLteam_lastmsgs}")

                            DSL_query_dict = extract_json_string(text = DSLteam_lastmsgs)
                            #print(f"DSL_query_dict: {DSL_query_dict}")
                            writing_status.update(label="DSL writing completed.", state="complete")
                        except Exception as e:
                            writing_status.update(label=f"DSL writing failed:{e}", state="error")

                elif st.session_state.field_select and (st.session_state.mode == "fast" or mode == "fast"):
                    with st.status("Selecting field & Writing DSL...", expanded=False) as DSL_status:
                        try:
                            DSL_query_dict = generate_elasticsearch_query_from_natural_language(query=requirement, description_path=os.path.join(mapping_dir, "filter_field_description.csv"),mapping_path=os.path.join(mapping_dir, "raw_mapping.json"))
                            with st.chat_message("ai"):
                                st.write(DSL_query_dict) 
                            DSL_status.update(label="Field selection & DSL writing completed.", state="complete")
                        except Exception as e:
                            DSL_status.update(label=f"Field selection & DSL writing failed:{e}", state="error")
                
                else:
                    DSL_query_dict = extract_json_string(text = HistoryTeam_lastmsgs)
                
                
                # OpenSearch and Save Retrived JSON 
                
                with st.status("Querying...", expanded=False) as Exe_status:
                    try:
                        ExeTeam_task = json.dumps(DSL_query_dict, indent=2, ensure_ascii=False)
                        Exeteam = await get_Exeteam(customer_name = st.session_state.customer_name)
                        Exe_stream = Exeteam.run_stream(task=ExeTeam_task)
                        async for Exe_msg in Exe_stream:
                            if not isinstance(Exe_msg, TextMessage):
                                continue
                            if Exe_msg.source == "user":
                                continue
                            st.session_state.messages.append({"role": Exe_msg.source, "content": Exe_msg.content})
                            with st.chat_message("ai"):
                                st.write_stream(stream_data(Exe_msg.content))  
                        
                        Exeteam_state = await Exeteam.save_state()
                        with open(os.path.join(team_state_dir, "Exeteam_state.json"), "w", encoding="utf-8") as f:
                            json.dump(Exeteam_state, f, ensure_ascii=False, indent=2, default=str)
                        Exeteam_msgs = Exeteam_state["agent_states"]["RoundRobinGroupChatManager"]["message_thread"]

                        for Exeteam_msg in reversed(Exeteam_msgs):
                            if Exeteam_msg.get("type") == "TextMessage":
                                if Exeteam_msg.get("source") == "DSLModifier":
                                    DSL_query_dict = extract_json_string(Exeteam_msg.get("content"))
                                    break
                                if Exeteam_msg.get("source") == "DSLExecutor":
                                    json_path_dict = extract_json_string(Exeteam_msg.get("content"))
                                    json_path_str = json.dumps(json_path_dict, ensure_ascii=False, indent=2).strip("\"'")

                        with target_dsl_path.open("w", encoding="utf-8") as f:
                            json.dump(DSL_query_dict, f, ensure_ascii=False, indent=2)

                        # Create Excel
                        excel_path, target_df = agg_json_to_excel(json_path=json_path_str)
                        st.session_state["excel_path"] = excel_path


                        excel_df = pd.read_excel(excel_path, engine="openpyxl")
                        st.write(excel_df)

                        # Create python script
                        write_export_script(
                            customer_name=st.session_state.customer_name,
                            excel_path="result.xlsx",
                            query=DSL_query_dict,
                            out_path=target_py_script,
                        )

                        st.session_state.finish = True
                        
                        Exe_status.update(label="Query successed.",state="complete", expanded=False)
                    except Exception as e:
                        Exe_status.update(label=f"Query failed:{e}.",state="error", expanded=False)          

        # DSL/EXCEL download
        if st.session_state.finish:
            # if prompt:

            target_df = pd.read_excel(st.session_state["excel_path"], engine='openpyxl')   
            with target_dsl_path.open("r", encoding="utf-8") as f:
                dsl = json.load(f)
            target_dsl = json.dumps(dsl, ensure_ascii=False, indent=2).encode("utf-8")

            excel_buffer = io.BytesIO()
            target_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label="Download Report",
                data=excel_buffer,
                file_name="report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon=":material/download:"
            )

            dsl_buffer = io.BytesIO()
            dsl_buffer.write(target_dsl)
            dsl_buffer.seek(0)
            st.download_button(
                label="Download DSL Query",
                data=dsl_buffer,
                file_name="DSL.json",
                mime="application/json",
                icon=":material/download:"
            )

            with open(target_py_script, "rb") as f:
                py_bytes = f.read()
            st.download_button(
                label="Download python script",
                data=py_bytes,
                file_name="opensearch_exporter.py",
                mime="text/x-python",
                icon=":material/download:",
            )


            # save report to DB: not history search(field_select == True) | history search but similar historical report not found(field_select == True)
            
            if st.session_state.field_select:
                SaveToDB = st.button("Save to Report DataBase", type="primary",key="save")
                if SaveToDB:
                    with st.status("Saving...", expanded=False) as save_status:
                        try:
                            ReportSaver = await get_ReportSaverTeam()
                            response = await ReportSaver.on_messages(
                                [TextMessage(content= target_dsl, source="user")],CancellationToken())           
                            
                            new_report = extract_json_string(response.chat_message.content)
                            
                            new_report["dsl"] = dsl
                            with st.chat_message("ai"):
                                st.write(new_report)

                            ReportSaver_state = await ReportSaver.save_state()
                            with open(os.path.join(team_state_dir, "ReportSaver_state.json"), "w", encoding="utf-8") as f:
                                json.dump(ReportSaver_state, f, ensure_ascii=False, indent=2, default=str)
                    
                    #======================================================================================================================
                            if st.session_state.brand.lower() == "huawei":
                                with open(os.path.join(rag_dir, "DSL_example_HUAWEI_copy.json"), "r", encoding="utf-8") as f:
                                    all_report = json.load(f)
                                all_report.append(new_report)                           
                                with open(os.path.join(rag_dir, "DSL_example_HUAWEI_copy.json"), "w", encoding="utf-8") as f:
                                    json.dump(all_report, f, ensure_ascii=False, indent=2)                        
                            elif st.session_state.brand.lower() == "aruba":
                                with open(os.path.join(rag_dir, "DSL_example_ARUBA_copy.json"), "r", encoding="utf-8") as f:
                                    all_report = json.load(f)                              
                                all_report.append(new_report)
                                with open(os.path.join(rag_dir, "DSL_example_ARUBA_copy.json"), "w", encoding="utf-8") as f:
                                    json.dump(all_report, f, ensure_ascii=False, indent=2)
                            save_status.update(label="✅ Save completed.", state="complete")
                        except Exception as e:
                            save_status.update(label=f"Save failed:{e}", state="error")
                        
    except Exception as e:
        st.write(f"error A:{e}")
        print(f"error A:{e}")
        # os.remove(os.path.join(team_state_dir,"Req_state.json"))

if __name__ == "__main__":
    asyncio.run(main())