/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:58:58 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.introspect.ClassIntrospector;
import com.fasterxml.jackson.databind.introspect.VirtualAnnotatedMember;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.POJONode;
import com.fasterxml.jackson.databind.node.ShortNode;
import com.fasterxml.jackson.databind.node.TextNode;
import java.io.IOException;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class UntypedObjectDeserializer_ESTest extends UntypedObjectDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      objectMapper0.readerForUpdating(untypedObjectDeserializer0);
      assertTrue(untypedObjectDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonDeserializer<?> jsonDeserializer0 = untypedObjectDeserializer0._withResolved(untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0);
      assertTrue(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonDeserializer<Object> jsonDeserializer0 = untypedObjectDeserializer0._clearIfStdImpl((JsonDeserializer<Object>) null);
      assertNull(jsonDeserializer0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      AnnotationMap annotationMap0 = new AnnotationMap();
      Class<Integer> class0 = Integer.class;
      AnnotationIntrospector annotationIntrospector0 = AnnotationIntrospector.nopInstance();
      AnnotatedClass annotatedClass0 = AnnotatedClass.construct(class0, annotationIntrospector0, (ClassIntrospector.MixInResolver) null);
      Class<Short> class1 = Short.class;
      VirtualAnnotatedMember virtualAnnotatedMember0 = new VirtualAnnotatedMember(annotatedClass0, class1, (String) null, class1);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      BeanProperty.Std beanProperty_Std0 = new BeanProperty.Std(propertyName0, (JavaType) null, propertyName0, annotationMap0, virtualAnnotatedMember0, propertyMetadata0);
      JsonDeserializer<?> jsonDeserializer0 = untypedObjectDeserializer0.createContextual(defaultDeserializationContext_Impl0, beanProperty_Std0);
      assertFalse(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      ObjectMapper objectMapper0 = new ObjectMapper();
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
  public void test05()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.insertObject(256);
      Long long0 = new Long(256);
      ObjectNode objectNode1 = objectNode0.put((String) null, long0);
      Double double0 = new Double(256);
      objectNode1.put("", double0);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(arrayNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
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
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(false);
      ObjectMapper objectMapper0 = new ObjectMapper();
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0.put("JSON", true);
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
      ObjectMapper objectMapper0 = new ObjectMapper();
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      POJONode pOJONode0 = new POJONode(untypedObjectDeserializer0);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(pOJONode0);
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
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      ObjectMapper objectMapper0 = new ObjectMapper();
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
  public void test09()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      ObjectMapper objectMapper0 = new ObjectMapper();
      TextNode textNode0 = new TextNode("[parameter #");
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(textNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0);
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
  public void test10()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ShortNode shortNode0 = ShortNode.valueOf((short)2);
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(shortNode0);
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
  public void test11()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      UntypedObjectDeserializer untypedObjectDeserializer1 = new UntypedObjectDeserializer(untypedObjectDeserializer0, untypedObjectDeserializer0, untypedObjectDeserializer_Vanilla0, untypedObjectDeserializer0, untypedObjectDeserializer0);
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      Double double0 = new Double(0.5076643200963111);
      ObjectNode objectNode1 = objectNode0.put((String) null, double0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(objectNode1);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
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
  public void test12()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.deserializeWithType(jsonParser0, (DeserializationContext) null, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      arrayNode0.insertObject(33);
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(arrayNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
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
  public void test14()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      objectNode0.putNull("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer");
      Double double0 = new Double(0.5076643200963111);
      ObjectNode objectNode1 = objectNode0.put((String) null, double0);
      ObjectMapper objectMapper0 = new ObjectMapper();
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
  public void test15()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer0.mapObject(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = new UntypedObjectDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.insertObject(3);
      Long long0 = new Long(1);
      ObjectNode objectNode1 = objectNode0.put("L|}", long0);
      Double double0 = new Double(1712.17395);
      objectNode1.put("co:.fasterxml.jackson.databind.deser.DefaultDeserializationContext$Impl", double0);
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
  public void test17()  throws Throwable  {
      UntypedObjectDeserializer untypedObjectDeserializer0 = UntypedObjectDeserializer.instance;
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectMapper objectMapper0 = new ObjectMapper();
      HashMap<String, JsonNode> hashMap0 = new HashMap<String, JsonNode>();
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0, hashMap0);
      ObjectNode objectNode1 = objectNode0.put("JSON", false);
      objectNode1.put("7k4ue=K", (short)2);
      Double double0 = new Double(33);
      objectNode0.put(">s/(i&ke7bTG", double0);
      objectNode0.put("I0K[gM~\"mv", double0);
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
  public void test18()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      // Undeclared exception!
      try { 
        untypedObjectDeserializer_Vanilla0.mapArray(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer$Vanilla", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
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
  public void test20()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonFactory jsonFactory0 = new JsonFactory(objectMapper0);
      char[] charArray0 = new char[8];
      charArray0[0] = '\"';
      charArray0[1] = '\"';
      charArray0[2] = '\"';
      charArray0[3] = '\"';
      charArray0[4] = '\"';
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = new UntypedObjectDeserializer.Vanilla();
      untypedObjectDeserializer_Vanilla0.mapObject(jsonParser0, deserializationContext0);
      try { 
        untypedObjectDeserializer_Vanilla0.mapArrayToArray(jsonParser0, deserializationContext0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Illegal unquoted character ((CTRL-CHAR, code 0)): has to be escaped using backslash to be included in string value
         //  at [Source: [C@0000000036; line: 1, column: 7]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }
}
