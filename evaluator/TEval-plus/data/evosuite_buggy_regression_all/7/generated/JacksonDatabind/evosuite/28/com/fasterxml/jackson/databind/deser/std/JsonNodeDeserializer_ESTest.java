/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:57:40 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.deser.std.JsonNodeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.DoubleNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonNodeDeserializer_ESTest extends JsonNodeDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Class<ArrayNode> class0 = ArrayNode.class;
      JsonDeserializer<? extends JsonNode> jsonDeserializer0 = JsonNodeDeserializer.getDeserializer(class0);
      assertTrue(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      JsonNodeDeserializer.ObjectDeserializer jsonNodeDeserializer_ObjectDeserializer0 = new JsonNodeDeserializer.ObjectDeserializer();
      assertTrue(jsonNodeDeserializer_ObjectDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      JsonNodeDeserializer jsonNodeDeserializer0 = new JsonNodeDeserializer();
      JsonNode jsonNode0 = jsonNodeDeserializer0.getNullValue((DeserializationContext) null);
      assertFalse(jsonNode0.isLong());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      JsonNodeDeserializer jsonNodeDeserializer0 = new JsonNodeDeserializer();
      JsonNode jsonNode0 = jsonNodeDeserializer0.getNullValue();
      assertEquals(JsonToken.VALUE_NULL, jsonNode0.asToken());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Class<ObjectNode> class0 = ObjectNode.class;
      JsonDeserializer<? extends JsonNode> jsonDeserializer0 = JsonNodeDeserializer.getDeserializer(class0);
      assertTrue(jsonDeserializer0.isCachable());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)91;
      Class<DoubleNode> class0 = DoubleNode.class;
      try { 
        objectMapper0.readValue(byteArray0, class0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unexpected character ('\uFFFD' (code 65533 / 0xfffd)): expected a valid value (number, String, array, object, 'true', 'false' or 'null')
         //  at [Source: [B@0000000013; line: 1, column: 3]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      JsonNodeDeserializer jsonNodeDeserializer0 = new JsonNodeDeserializer();
      // Undeclared exception!
      try { 
        jsonNodeDeserializer0.deserialize(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.JsonNodeDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      JsonNodeDeserializer.ObjectDeserializer jsonNodeDeserializer_ObjectDeserializer0 = JsonNodeDeserializer.ObjectDeserializer._instance;
      // Undeclared exception!
      try { 
        jsonNodeDeserializer_ObjectDeserializer0.deserialize(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.JsonNodeDeserializer$ObjectDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      JsonNodeDeserializer.ArrayDeserializer jsonNodeDeserializer_ArrayDeserializer0 = new JsonNodeDeserializer.ArrayDeserializer();
      // Undeclared exception!
      try { 
        jsonNodeDeserializer_ArrayDeserializer0.deserialize(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.JsonNodeDeserializer$ArrayDeserializer", e);
      }
  }
}