/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:35:13 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.base.GeneratorBase;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonStringFormatVisitor;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.StdKeySerializer;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import java.io.OutputStream;
import java.lang.reflect.Type;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdKeySerializer_ESTest extends StdKeySerializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      StdKeySerializer stdKeySerializer0 = new StdKeySerializer();
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonNode jsonNode0 = stdKeySerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) simpleType0);
      assertFalse(jsonNode0.isDouble());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      StdKeySerializer stdKeySerializer0 = new StdKeySerializer();
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonStringFormatVisitor jsonStringFormatVisitor0 = mock(JsonStringFormatVisitor.class, new ViolatedAssumptionAnswer());
      JsonFormatVisitorWrapper jsonFormatVisitorWrapper0 = mock(JsonFormatVisitorWrapper.class, new ViolatedAssumptionAnswer());
      doReturn(jsonStringFormatVisitor0).when(jsonFormatVisitorWrapper0).expectStringFormat(any(com.fasterxml.jackson.databind.JavaType.class));
      stdKeySerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper0, simpleType0);
      assertTrue(simpleType0.isConcrete());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      StdKeySerializer stdKeySerializer0 = new StdKeySerializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonFactory jsonFactory0 = new JsonFactory(objectMapper0);
      UTF8JsonGenerator uTF8JsonGenerator0 = (UTF8JsonGenerator)jsonFactory0.createGenerator((OutputStream) null);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      stdKeySerializer0.serialize(defaultSerializerProvider_Impl0, uTF8JsonGenerator0, defaultSerializerProvider_Impl0);
      assertEquals(55296, GeneratorBase.SURR1_FIRST);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      StdKeySerializer stdKeySerializer0 = new StdKeySerializer();
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonFactory jsonFactory0 = new JsonFactory(objectMapper0);
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) null);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      MockDate mockDate0 = new MockDate(56320, 57343, 55296);
      // Undeclared exception!
      try { 
        stdKeySerializer0.serialize(mockDate0, jsonGenerator0, defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }
}