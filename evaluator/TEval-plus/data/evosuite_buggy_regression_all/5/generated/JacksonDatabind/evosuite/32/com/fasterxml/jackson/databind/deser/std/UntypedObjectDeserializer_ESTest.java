/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:02:22 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.BooleanNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.NumericNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.io.PipedReader;
import java.io.Reader;
import java.io.StringReader;
import java.util.HashMap;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class UntypedObjectDeserializer_ESTest extends UntypedObjectDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      ArrayNode arrayNode0 = objectNode0.withArray("");
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(arrayNode0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.readerForUpdating(untypedObjectDeserializer0);
      assertTrue(untypedObjectDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      JsonDeserializer<Object> jsonDeserializer0 = untypedObjectDeserializer0._clearIfStdImpl((JsonDeserializer<Object>) null);
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      JsonDeserializer<?> jsonDeserializer0 = untypedObjectDeserializer1.createContextual(deserializationContext0, (BeanProperty) null);
      assertTrue(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      JsonDeserializer<?> jsonDeserializer0 = untypedObjectDeserializer0.createContextual(deserializationContext0, (BeanProperty) null);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      ArrayNode arrayNode0 = objectNode0.withArray("");
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(arrayNode0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0.putNull("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer");
      objectNode0.withArray((String) null);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ArrayNode arrayNode0 = objectNode0.withArray((String) null);
      arrayNode0.add("w%+;2uO'x==e");
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(true);
      NumericNode numericNode0 = jsonNodeFactory0.numberNode((double) (byte)32);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(numericNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArrayToArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      BooleanNode booleanNode0 = BooleanNode.valueOf(true);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(booleanNode0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArrayToArray(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
      Byte byte0 = new Byte((byte)64);
      UntypedObjectDeserializer untypedObjectDeserializer2 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer1, untypedObjectDeserializer_Vanilla0);
      untypedObjectDeserializer0._withResolved(untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer1, untypedObjectDeserializer1, untypedObjectDeserializer1);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      objectNode0.put((String) null, (int) (byte)64);
      Vector<Module> vector0 = new Vector<Module>((byte)64);
      ObjectMapper objectMapper1 = objectMapper0.registerModules((Iterable<Module>) vector0);
      JsonParser jsonParser0 = objectMapper1.treeAsTokens(objectNode0);
      DeserializationContext deserializationContext0 = objectMapper1.getDeserializationContext();
      UntypedObjectDeserializer untypedObjectDeserializer3 = new UntypedObjectDeserializer();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArrayToArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonFactory jsonFactory0 = new JsonFactory();
      StringReader stringReader0 = new StringReader("JSON");
      JsonParser jsonParser0 = jsonFactory0.createParser((Reader) stringReader0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.deserializeWithType(jsonParser0, deserializationContext0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ArrayNode arrayNode0 = objectNode0.withArray((String) null);
      arrayNode0.add("w%+;2uO'x==e");
      arrayNode0.add("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer$Vanilla");
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      Byte byte0 = new Byte((byte)0);
      JsonDeserializer<Byte> jsonDeserializer0 = (JsonDeserializer<Byte>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      doReturn(byte0, byte0, byte0).when(jsonDeserializer0).deserialize(any(com.fasterxml.jackson.core.JsonParser.class) , any(com.fasterxml.jackson.databind.DeserializationContext.class));
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, jsonDeserializer0, jsonDeserializer0, jsonDeserializer0, untypedObjectDeserializer0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      objectNode0.withArray((String) null);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      untypedObjectDeserializer1.mapArray(jsonParser0, deserializationContext0);
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      // Undeclared exception!
      try { 
        untypedObjectDeserializer_Vanilla0.deserialize(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonFactory jsonFactory0 = new JsonFactory();
      StringReader stringReader0 = new StringReader("JSON");
      JsonParser jsonParser0 = jsonFactory0.createParser((Reader) stringReader0);
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      try { 
        untypedObjectDeserializer0.mapObject(jsonParser0, deserializationContext0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Can not deserialize instance of java.lang.Object out of null token
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      ObjectNode objectNode1 = objectNode0.put("JSON", true);
      byte[] byteArray0 = new byte[3];
      objectNode0.put("", byteArray0);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode1);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      ObjectNode objectNode1 = objectNode0.put("JSON", true);
      byte[] byteArray0 = new byte[3];
      objectNode1.put("", byteArray0);
      objectNode1.withArray("MU|>l+xqMz_}");
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode1);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      ObjectNode objectNode1 = objectNode0.put("", false);
      Short short0 = new Short((byte)10);
      objectNode1.put("0<uX", short0);
      Double double0 = new Double((-95.089967674));
      ObjectNode objectNode2 = objectNode0.put("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer", (short)679);
      objectNode2.put("com.fasterxml.jackson.annotation.ObjectIdGenerators$Base", double0);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapArrayToArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.insertObject((byte)32);
      ObjectNode objectNode1 = objectNode0.with("/pom.properties");
      objectNode1.withArray((String) null);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Object[] objectArray0 = untypedObjectDeserializer0.mapArrayToArray(jsonParser0, deserializationContext0);
      assertEquals(4, jsonParser0.getCurrentTokenId());
      assertEquals(1, objectArray0.length);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      objectNode0.putNull((String) null);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArrayToArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonFactory jsonFactory0 = new JsonFactory();
      StringReader stringReader0 = new StringReader("JSON");
      JsonParser jsonParser0 = jsonFactory0.createParser((Reader) stringReader0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      // Undeclared exception!
      try { 
        untypedObjectDeserializer_Vanilla0.mapArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonFactory jsonFactory0 = new JsonFactory();
      PipedReader pipedReader0 = new PipedReader();
      JsonParser jsonParser0 = jsonFactory0.createParser((Reader) pipedReader0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      // Undeclared exception!
      try { 
        untypedObjectDeserializer_Vanilla0.deserializeWithType(jsonParser0, deserializationContext0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      ObjectNode objectNode1 = objectNode0.putNull((String) null);
      objectNode1.withArray("");
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode1);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArrayToArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      ObjectNode objectNode1 = objectNode0.putNull((String) null);
      objectNode1.withArray("");
      Short short0 = new Short((byte)64);
      ObjectNode objectNode2 = objectNode1.put("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer", short0);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode2);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArrayToArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectNode objectNode0 = objectMapper0.createObjectNode();
      objectNode0.withArray(")");
      ObjectNode objectNode1 = objectNode0.put("", true);
      Float float0 = new Float(143.31854F);
      objectNode1.put("v0YK&?C", float0);
      objectNode1.with("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer");
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode1);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0);
      // Undeclared exception!
      try { 
        untypedObjectDeserializer1.mapArrayToArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }
}