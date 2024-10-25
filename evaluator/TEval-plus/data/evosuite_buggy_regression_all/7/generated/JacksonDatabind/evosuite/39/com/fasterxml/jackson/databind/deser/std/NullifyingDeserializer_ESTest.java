/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:00:22 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.deser.std.NullifyingDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NullifyingDeserializer_ESTest extends NullifyingDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      NullifyingDeserializer nullifyingDeserializer0 = new NullifyingDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Object object0 = nullifyingDeserializer0.deserialize(jsonParser0, deserializationContext0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      NullifyingDeserializer nullifyingDeserializer0 = new NullifyingDeserializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ArrayNode arrayNode0 = objectMapper0.createArrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse((ObjectCodec) objectMapper0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Object object0 = nullifyingDeserializer0.deserializeWithType(jsonParser0, deserializationContext0, (TypeDeserializer) null);
      assertNull(object0);
  }
}
