/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:26:57 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ext.CoreXMLSerializers;
import com.fasterxml.jackson.databind.ext.DOMSerializer;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsontype.TypeSerializer;
import com.fasterxml.jackson.databind.node.FloatNode;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.std.StdDelegatingSerializer;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.Converter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdDelegatingSerializer_ESTest extends StdDelegatingSerializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<ObjectReader> class0 = ObjectReader.class;
      Converter<ObjectReader, CoreXMLSerializers.XMLGregorianCalendarSerializer> converter0 = (Converter<ObjectReader, CoreXMLSerializers.XMLGregorianCalendarSerializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(converter0).toString();
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(class0, (Converter<ObjectReader, ?>) converter0);
      Converter<Object, ?> converter1 = stdDelegatingSerializer0.getConverter();
      assertNotNull(converter1);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<BeanSerializer> class0 = BeanSerializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      Converter<Object, Integer> converter0 = (Converter<Object, Integer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(converter0).convert(any());
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(converter0, simpleType0, defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      FloatNode floatNode0 = FloatNode.valueOf(0L);
      try { 
        stdDelegatingSerializer0.serializeWithType(floatNode0, (JsonGenerator) null, defaultSerializerProvider_Impl0, (TypeSerializer) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Type id handling not implemented for type java.lang.Object (by serializer of type com.fasterxml.jackson.databind.ser.impl.FailingSerializer)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer((Converter<?, ?>) null);
      JsonSerializer<?> jsonSerializer0 = stdDelegatingSerializer0.getDelegatee();
      assertNull(jsonSerializer0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<String> class0 = String.class;
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(class0, (Converter<String, ?>) null);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        stdDelegatingSerializer0.isEmpty((SerializerProvider) defaultSerializerProvider_Impl0, (Object) defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdDelegatingSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer((Converter<?, ?>) null);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base(defaultSerializerProvider_Impl0);
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = SimpleType.construct(class0);
      // Undeclared exception!
      try { 
        stdDelegatingSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, simpleType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdDelegatingSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<ChronoUnit> class0 = ChronoUnit.class;
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(class0, (Converter<ChronoUnit, ?>) null);
      ChronoUnit chronoUnit0 = ChronoUnit.HALF_DAYS;
      // Undeclared exception!
      try { 
        stdDelegatingSerializer0.isEmpty((Object) chronoUnit0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdDelegatingSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<ObjectReader> class0 = ObjectReader.class;
      Converter<ObjectReader, CoreXMLSerializers.XMLGregorianCalendarSerializer> converter0 = (Converter<ObjectReader, CoreXMLSerializers.XMLGregorianCalendarSerializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(class0, (Converter<ObjectReader, ?>) converter0);
      stdDelegatingSerializer0.resolve((SerializerProvider) null);
      assertFalse(stdDelegatingSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Converter<Object, DOMSerializer> converter0 = (Converter<Object, DOMSerializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      Class<ObjectReader> class0 = ObjectReader.class;
      Converter<ObjectReader, CoreXMLSerializers.XMLGregorianCalendarSerializer> converter1 = (Converter<ObjectReader, CoreXMLSerializers.XMLGregorianCalendarSerializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(class0, (Converter<ObjectReader, ?>) converter1);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class1 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class1);
      JsonSerializer<BeanSerializer> jsonSerializer0 = (JsonSerializer<BeanSerializer>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      StdDelegatingSerializer stdDelegatingSerializer1 = stdDelegatingSerializer0.withDelegate(converter0, mapType0, jsonSerializer0);
      stdDelegatingSerializer1.resolve((SerializerProvider) null);
      assertFalse(stdDelegatingSerializer1.isUnwrappingSerializer());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<ObjectReader> class0 = ObjectReader.class;
      Converter<ObjectReader, CoreXMLSerializers.XMLGregorianCalendarSerializer> converter0 = (Converter<ObjectReader, CoreXMLSerializers.XMLGregorianCalendarSerializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(class0, (Converter<ObjectReader, ?>) converter0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class1 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class1);
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(mapType0);
      Class<BeanSerializer> class2 = BeanSerializer.class;
      SimpleType simpleType0 = SimpleType.construct(class2);
      Converter<Object, String> converter1 = (Converter<Object, String>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingSerializer stdDelegatingSerializer1 = stdDelegatingSerializer0.withDelegate(converter1, simpleType0, beanSerializer0);
      stdDelegatingSerializer1.resolve((SerializerProvider) null);
      assertFalse(stdDelegatingSerializer1.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class1 = HashMap.class;
      MapType mapType0 = typeFactory0.constructMapType((Class<? extends Map>) class1, (JavaType) simpleType0, (JavaType) simpleType0);
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer((Converter<Object, ?>) null, mapType0, (JsonSerializer<?>) null);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        stdDelegatingSerializer0.createContextual(defaultSerializerProvider_Impl0, (BeanProperty) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<ChronoUnit> class0 = ChronoUnit.class;
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(class0, (Converter<ChronoUnit, ?>) null);
      // Undeclared exception!
      try { 
        stdDelegatingSerializer0.createContextual(defaultSerializerProvider_Impl0, (BeanProperty) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Converter<Object, Integer> converter0 = (Converter<Object, Integer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      Class<StdDelegatingSerializer> class0 = StdDelegatingSerializer.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      JsonSerializer<Integer> jsonSerializer0 = (JsonSerializer<Integer>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(converter0, simpleType0, jsonSerializer0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Converter<Object, CoreXMLSerializers.XMLGregorianCalendarSerializer> converter1 = (Converter<Object, CoreXMLSerializers.XMLGregorianCalendarSerializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingSerializer stdDelegatingSerializer1 = new StdDelegatingSerializer(converter1, simpleType0, stdDelegatingSerializer0);
      JsonSerializer<?> jsonSerializer1 = stdDelegatingSerializer1.createContextual(defaultSerializerProvider_Impl0, (BeanProperty) null);
      assertFalse(jsonSerializer1.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      Class<Integer> class0 = Integer.class;
      ObjectReader objectReader0 = objectMapper0.reader((Class<?>) class0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      Class<Object> class1 = Object.class;
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class1);
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(mapLikeType0);
      Converter<ObjectWriter, BeanSerializer> converter0 = (Converter<ObjectWriter, BeanSerializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      doReturn(beanSerializer0).when(converter0).convert(any(com.fasterxml.jackson.databind.ObjectWriter.class));
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(converter0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        stdDelegatingSerializer0.serialize((Object) null, (JsonGenerator) null, defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdDelegatingSerializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Converter<ObjectWriter, BeanSerializer> converter0 = (Converter<ObjectWriter, BeanSerializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(converter0).convert(any(com.fasterxml.jackson.databind.ObjectWriter.class));
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(converter0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      // Undeclared exception!
      try { 
        stdDelegatingSerializer0.serialize((Object) null, (JsonGenerator) null, defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Converter<String, String> converter0 = (Converter<String, String>) mock(Converter.class, new ViolatedAssumptionAnswer());
      Class<String> class0 = String.class;
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(class0, (Converter<String, ?>) converter0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonNode jsonNode0 = stdDelegatingSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0);
      assertFalse(jsonNode0.isShort());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<String> class0 = String.class;
      Class<ChronoUnit> class1 = ChronoUnit.class;
      SimpleType simpleType0 = SimpleType.construct(class1);
      MapType mapType0 = MapType.construct(class0, simpleType0, simpleType0);
      DOMSerializer dOMSerializer0 = new DOMSerializer();
      Class<Object> class2 = Object.class;
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(class2, (Converter<Object, ?>) null);
      StdDelegatingSerializer stdDelegatingSerializer1 = stdDelegatingSerializer0.withDelegate((Converter<Object, ?>) null, mapType0, dOMSerializer0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonNode jsonNode0 = stdDelegatingSerializer1.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) mapType0);
      assertNull(jsonNode0.textValue());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Class<ChronoUnit> class0 = ChronoUnit.class;
      Converter<ChronoUnit, ChronoUnit> converter0 = (Converter<ChronoUnit, ChronoUnit>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(class0, (Converter<ChronoUnit, ?>) converter0);
      JsonNode jsonNode0 = stdDelegatingSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) class0, true);
      assertFalse(jsonNode0.isFloatingPointNumber());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Converter<StdDelegatingSerializer, String> converter0 = (Converter<StdDelegatingSerializer, String>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingSerializer stdDelegatingSerializer0 = new StdDelegatingSerializer(converter0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      Converter<Object, DOMSerializer> converter1 = (Converter<Object, DOMSerializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<String> class0 = String.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      StdDelegatingSerializer stdDelegatingSerializer1 = stdDelegatingSerializer0.withDelegate(converter1, mapLikeType0, defaultSerializerProvider_Impl0.DEFAULT_NULL_KEY_SERIALIZER);
      // Undeclared exception!
      try { 
        stdDelegatingSerializer1.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) mapLikeType0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.StdSerializer", e);
      }
  }
}
