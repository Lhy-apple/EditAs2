/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:20:12 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.NullNode;
import java.math.BigDecimal;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StringArrayDeserializer_ESTest extends StringArrayDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        stringArrayDeserializer0.instance.deserializeWithType(jsonParser0, deserializationContext0, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        stringArrayDeserializer0.deserialize(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      StringArrayDeserializer stringArrayDeserializer2 = new StringArrayDeserializer(stringArrayDeserializer1);
      // Undeclared exception!
      try { 
        stringArrayDeserializer2._deserializeCustom(jsonParser0, deserializationContext0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // [Ljava.lang.String; cannot be cast to java.lang.String
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      BigDecimal bigDecimal0 = BigDecimal.TEN;
      ArrayNode arrayNode1 = arrayNode0.add(bigDecimal0);
      JsonParser jsonParser0 = arrayNode1.traverse();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer1._deserializeCustom(jsonParser0, deserializationContext0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // [Ljava.lang.String; cannot be cast to java.lang.String
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      ArrayNode arrayNode1 = arrayNode0.add((BigDecimal) null);
      JsonParser jsonParser0 = arrayNode1.traverse();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      StringArrayDeserializer stringArrayDeserializer1 = new StringArrayDeserializer(stringArrayDeserializer0);
      // Undeclared exception!
      try { 
        stringArrayDeserializer1._deserializeCustom(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = new StringArrayDeserializer();
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      NullNode nullNode0 = NullNode.instance;
      JsonParser jsonParser0 = objectMapper0.treeAsTokens(nullNode0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        stringArrayDeserializer0._deserializeCustom(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      StringArrayDeserializer stringArrayDeserializer0 = StringArrayDeserializer.instance;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        stringArrayDeserializer0._deserializeCustom(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StringArrayDeserializer", e);
      }
  }
}