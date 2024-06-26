/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:59:07 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.base.GeneratorBase;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.WriterBasedJsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonStringFormatVisitor;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.StdKeySerializer;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.io.FilterOutputStream;
import java.io.OutputStreamWriter;
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
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonNode jsonNode0 = stdKeySerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) simpleType0);
      assertNull(jsonNode0.textValue());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      StdKeySerializer stdKeySerializer0 = new StdKeySerializer();
      JsonFormatVisitorWrapper jsonFormatVisitorWrapper0 = mock(JsonFormatVisitorWrapper.class, new ViolatedAssumptionAnswer());
      doReturn((JsonStringFormatVisitor) null).when(jsonFormatVisitorWrapper0).expectStringFormat(any(com.fasterxml.jackson.databind.JavaType.class));
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Integer> class0 = Integer.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      stdKeySerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper0, mapLikeType0);
      assertFalse(stdKeySerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      StdKeySerializer stdKeySerializer0 = new StdKeySerializer();
      Object object0 = new Object();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(bufferRecycler0, 0);
      FilterOutputStream filterOutputStream0 = new FilterOutputStream(byteArrayBuilder0);
      OutputStreamWriter outputStreamWriter0 = new OutputStreamWriter(filterOutputStream0);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 0, (ObjectCodec) null, outputStreamWriter0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      stdKeySerializer0.serialize(object0, writerBasedJsonGenerator0, defaultSerializerProvider_Impl0);
      assertEquals(57343, GeneratorBase.SURR2_LAST);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      StdKeySerializer stdKeySerializer0 = new StdKeySerializer();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(bufferRecycler0, 0);
      FilterOutputStream filterOutputStream0 = new FilterOutputStream(byteArrayBuilder0);
      OutputStreamWriter outputStreamWriter0 = new OutputStreamWriter(filterOutputStream0);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 0, (ObjectCodec) null, outputStreamWriter0);
      MockDate mockDate0 = new MockDate(57343, 3, 3);
      // Undeclared exception!
      try { 
        stdKeySerializer0.serialize(mockDate0, writerBasedJsonGenerator0, (SerializerProvider) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdKeySerializer", e);
      }
  }
}
