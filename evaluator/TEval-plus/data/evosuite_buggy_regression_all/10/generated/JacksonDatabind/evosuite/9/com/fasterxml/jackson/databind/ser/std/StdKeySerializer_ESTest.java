/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:24:39 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonStringFormatVisitor;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.StdKeySerializer;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.File;
import java.lang.reflect.Type;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdKeySerializer_ESTest extends StdKeySerializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      StdKeySerializer stdKeySerializer0 = new StdKeySerializer();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Map> class0 = Map.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      JsonNode jsonNode0 = stdKeySerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) mapType0);
      assertFalse(jsonNode0.isLong());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      StdKeySerializer stdKeySerializer0 = new StdKeySerializer();
      JsonFormatVisitorWrapper jsonFormatVisitorWrapper0 = mock(JsonFormatVisitorWrapper.class, new ViolatedAssumptionAnswer());
      doReturn((JsonStringFormatVisitor) null).when(jsonFormatVisitorWrapper0).expectStringFormat(any(com.fasterxml.jackson.databind.JavaType.class));
      stdKeySerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper0, (JavaType) null);
      assertFalse(stdKeySerializer0.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      StdKeySerializer stdKeySerializer0 = new StdKeySerializer();
      JsonFactory jsonFactory0 = new JsonFactory();
      File file0 = MockFile.createTempFile("JSON", "JSON");
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF8;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator(file0, jsonEncoding0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      stdKeySerializer0.serialize(jsonEncoding0, jsonGenerator0, defaultSerializerProvider_Impl0);
      assertFalse(jsonGenerator0.isClosed());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      StdKeySerializer stdKeySerializer0 = new StdKeySerializer();
      JsonFactory jsonFactory0 = new JsonFactory();
      File file0 = MockFile.createTempFile("': no back reference property found from type ", "': no back reference property found from type ");
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF8;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator(file0, jsonEncoding0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      MockDate mockDate0 = new MockDate((-172), (-172), 53);
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
