/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:59:52 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.deser.std.NullifyingDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NullifyingDeserializer_ESTest extends NullifyingDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      NullifyingDeserializer nullifyingDeserializer0 = new NullifyingDeserializer();
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      Object object0 = nullifyingDeserializer0.deserialize(jsonParser0, (DeserializationContext) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      NullifyingDeserializer nullifyingDeserializer0 = NullifyingDeserializer.instance;
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      Object object0 = nullifyingDeserializer0.deserializeWithType(jsonParser0, (DeserializationContext) null, (TypeDeserializer) null);
      assertNull(object0);
  }
}