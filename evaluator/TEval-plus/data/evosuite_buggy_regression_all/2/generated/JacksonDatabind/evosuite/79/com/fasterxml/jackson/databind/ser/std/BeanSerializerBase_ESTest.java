/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:04:43 GMT 2023
 */

package com.fasterxml.jackson.databind.ser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.json.WriterBasedJsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.core.util.JsonGeneratorDelegate;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.cfg.MapperConfig;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.introspect.BasicBeanDescription;
import com.fasterxml.jackson.databind.introspect.ObjectIdInfo;
import com.fasterxml.jackson.databind.introspect.POJOPropertiesCollector;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonFormatVisitorWrapper;
import com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeSerializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.ser.BeanPropertyWriter;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.BeanSerializerBuilder;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.ser.PropertyWriter;
import com.fasterxml.jackson.databind.ser.impl.ObjectIdWriter;
import com.fasterxml.jackson.databind.ser.std.BeanSerializerBase;
import com.fasterxml.jackson.databind.ser.std.StdArraySerializers;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.NameTransformer;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PipedOutputStream;
import java.io.Writer;
import java.lang.reflect.Type;
import java.util.Iterator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.evosuite.runtime.mock.java.io.MockPrintWriter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanSerializerBase_ESTest extends BeanSerializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanSerializerBase beanSerializerBase0 = beanSerializer0.withFilterId(defaultSerializerProvider_Impl0);
      // Undeclared exception!
      try { 
        beanSerializerBase0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) javaType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      Iterator<PropertyWriter> iterator0 = beanSerializer0.properties();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      NameTransformer nameTransformer0 = NameTransformer.NOP;
      JsonSerializer<Object> jsonSerializer0 = beanSerializer0.unwrappingSerializer(nameTransformer0);
      assertFalse(jsonSerializer0.usesObjectId());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(bufferRecycler0, 52);
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(byteArrayBuilder0);
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, (DefaultDeserializationContext) null);
      BufferedWriter bufferedWriter0 = new BufferedWriter(mockPrintWriter0);
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forSerialization(pOJOPropertiesCollector0);
      try { 
        objectMapper0.writeValue((Writer) bufferedWriter0, (Object) basicBeanDescription0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // (was java.lang.NullPointerException) (through reference chain: com.fasterxml.jackson.databind.introspect.BasicBeanDescription[\"classAnnotations\"])
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter(pipedOutputStream0);
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, (DefaultDeserializationContext) null);
      BufferedWriter bufferedWriter0 = new BufferedWriter(mockPrintWriter0);
      try { 
        objectMapper0.writeValue((Writer) bufferedWriter0, (Object) jsonFactory0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Infinite recursion (StackOverflowError) (through reference chain: com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"]->com.fasterxml.jackson.core.JsonFactory[\"codec\"]->com.fasterxml.jackson.databind.ObjectMapper[\"factory\"])
         //
         verifyException("com.fasterxml.jackson.databind.ser.std.BeanSerializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      boolean boolean0 = beanSerializer0.usesObjectId();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      POJOPropertiesCollector pOJOPropertiesCollector0 = mock(POJOPropertiesCollector.class, new ViolatedAssumptionAnswer());
      doReturn((AnnotatedClass) null).when(pOJOPropertiesCollector0).getClassDef();
      doReturn((MapperConfig) null).when(pOJOPropertiesCollector0).getConfig();
      doReturn((ObjectIdInfo) null).when(pOJOPropertiesCollector0).getObjectIdInfo();
      doReturn((JavaType) null).when(pOJOPropertiesCollector0).getType();
      BasicBeanDescription basicBeanDescription0 = BasicBeanDescription.forDeserialization(pOJOPropertiesCollector0);
      BeanSerializerBuilder beanSerializerBuilder0 = new BeanSerializerBuilder(basicBeanDescription0);
      BeanSerializer beanSerializer0 = new BeanSerializer(javaType0, beanSerializerBuilder0, (BeanPropertyWriter[]) null, (BeanPropertyWriter[]) null);
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      ObjectIdGenerator<StdArraySerializers.ShortArraySerializer> objectIdGenerator0 = (ObjectIdGenerator<StdArraySerializers.ShortArraySerializer>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      ObjectIdWriter objectIdWriter0 = ObjectIdWriter.construct(javaType0, propertyName0, objectIdGenerator0, false);
      BeanSerializerBase beanSerializerBase0 = beanSerializer0.withObjectIdWriter(objectIdWriter0);
      boolean boolean0 = beanSerializerBase0.usesObjectId();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      ObjectMapper objectMapper0 = new ObjectMapper();
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, objectMapper0, true);
      WriterBasedJsonGenerator writerBasedJsonGenerator0 = new WriterBasedJsonGenerator(iOContext0, 2, objectMapper0, (Writer) null);
      JsonGeneratorDelegate jsonGeneratorDelegate0 = new JsonGeneratorDelegate(writerBasedJsonGenerator0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(javaType0, typeFactory0);
      AsArrayTypeSerializer asArrayTypeSerializer0 = new AsArrayTypeSerializer(classNameIdResolver0, (BeanProperty) null);
      beanSerializer0.serializeWithType(javaType0, jsonGeneratorDelegate0, serializerProvider0, asArrayTypeSerializer0);
      assertEquals(53, jsonGeneratorDelegate0.getOutputBuffered());
      assertEquals(53, writerBasedJsonGenerator0.getOutputBuffered());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, (DefaultDeserializationContext) null);
      MockPrintStream mockPrintStream0 = new MockPrintStream(pipedOutputStream0, false);
      IOContext iOContext0 = new IOContext(bufferRecycler0, mockPrintStream0, true);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 1, objectMapper0, mockPrintStream0);
      // Undeclared exception!
      try { 
        ((BeanSerializerBase)beanSerializer0).serializeFieldsFiltered(defaultSerializerProvider_Impl0, uTF8JsonGenerator0, defaultSerializerProvider_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.SerializerProvider", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      JsonNode jsonNode0 = beanSerializer0.getSchema((SerializerProvider) defaultSerializerProvider_Impl0, (Type) javaType0);
      assertEquals(2, jsonNode0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      JsonFormatVisitorWrapper.Base jsonFormatVisitorWrapper_Base0 = new JsonFormatVisitorWrapper.Base();
      beanSerializer0.acceptJsonFormatVisitor(jsonFormatVisitorWrapper_Base0, javaType0);
      assertFalse(javaType0.isMapLikeType());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      BeanSerializer beanSerializer0 = BeanSerializer.createDummy(javaType0);
      beanSerializer0.acceptJsonFormatVisitor((JsonFormatVisitorWrapper) null, javaType0);
      assertFalse(beanSerializer0.isUnwrappingSerializer());
  }
}