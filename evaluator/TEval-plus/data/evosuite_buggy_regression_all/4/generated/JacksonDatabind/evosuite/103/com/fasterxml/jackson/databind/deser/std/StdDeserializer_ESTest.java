/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:49:52 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.filter.FilteringParserDelegate;
import com.fasterxml.jackson.core.filter.TokenFilter;
import com.fasterxml.jackson.core.io.InputDecorator;
import com.fasterxml.jackson.core.util.JsonParserSequence;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.DateDeserializers;
import com.fasterxml.jackson.databind.deser.std.FactoryBasedEnumDeserializer;
import com.fasterxml.jackson.databind.deser.std.JsonNodeDeserializer;
import com.fasterxml.jackson.databind.deser.std.NumberDeserializers;
import com.fasterxml.jackson.databind.deser.std.PrimitiveArrayDeserializers;
import com.fasterxml.jackson.databind.deser.std.StdDelegatingDeserializer;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.PlaceholderForType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.util.AccessPattern;
import com.fasterxml.jackson.databind.util.Converter;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdDeserializer_ESTest extends StdDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PrimitiveArrayDeserializers.ByteDeser primitiveArrayDeserializers_ByteDeser0 = new PrimitiveArrayDeserializers.ByteDeser();
      String string0 = primitiveArrayDeserializers_ByteDeser0._coercedTypeDesc();
      assertEquals("as content of type `byte[]`", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      NumberDeserializers.LongDeserializer numberDeserializers_LongDeserializer0 = NumberDeserializers.LongDeserializer.primitiveInstance;
      PlaceholderForType placeholderForType0 = new PlaceholderForType((-359));
      Class<Integer> class0 = Integer.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
      ArrayType arrayType0 = ArrayType.construct((JavaType) placeholderForType0, typeBindings0);
      StdDelegatingDeserializer<Object> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<Object>((Converter<Object, Object>) null, arrayType0, numberDeserializers_LongDeserializer0);
      StdDelegatingDeserializer<Object> stdDelegatingDeserializer1 = new StdDelegatingDeserializer<Object>(stdDelegatingDeserializer0);
      assertEquals(AccessPattern.DYNAMIC, stdDelegatingDeserializer1.getEmptyAccessPattern());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      NumberDeserializers.ByteDeserializer numberDeserializers_ByteDeserializer0 = NumberDeserializers.ByteDeserializer.wrapperInstance;
      JsonFactory jsonFactory0 = new JsonFactory();
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false, false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      InputDecorator inputDecorator0 = mock(InputDecorator.class, new ViolatedAssumptionAnswer());
      doReturn(sequenceInputStream0).when(inputDecorator0).decorate(any(com.fasterxml.jackson.core.io.IOContext.class) , any(byte[].class) , anyInt() , anyInt());
      jsonFactory0.setInputDecorator(inputDecorator0);
      byte[] byteArray0 = new byte[4];
      JsonParser jsonParser0 = jsonFactory0.createParser(byteArray0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        numberDeserializers_ByteDeserializer0._verifyEndArrayForSingle(jsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Converter<Integer, Short> converter0 = (Converter<Integer, Short>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<Short> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<Short>(converter0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      BeanProperty.Bogus beanProperty_Bogus0 = new BeanProperty.Bogus();
      JavaType javaType0 = beanProperty_Bogus0.getType();
      // Undeclared exception!
      try { 
        stdDelegatingDeserializer0.findDeserializer(defaultDeserializationContext_Impl0, javaType0, beanProperty_Bogus0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.DeserializerCache", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StdDelegatingDeserializer<BeanDeserializer> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<BeanDeserializer>((Converter<?, BeanDeserializer>) null);
      Class<?> class0 = stdDelegatingDeserializer0.getValueClass();
      assertFalse(class0.isEnum());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JsonNodeDeserializer.ObjectDeserializer jsonNodeDeserializer_ObjectDeserializer0 = new JsonNodeDeserializer.ObjectDeserializer();
      Converter<String, BeanDeserializer> converter0 = (Converter<String, BeanDeserializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<BeanDeserializer> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<BeanDeserializer>(converter0);
      boolean boolean0 = stdDelegatingDeserializer0.isDefaultDeserializer(jsonNodeDeserializer_ObjectDeserializer0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      DateDeserializers.CalendarDeserializer dateDeserializers_CalendarDeserializer0 = new DateDeserializers.CalendarDeserializer();
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate((JsonParser) null, tokenFilter0, false, false);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(false, (JsonParser) filteringParserDelegate0, (JsonParser) null);
      // Undeclared exception!
      try { 
        dateDeserializers_CalendarDeserializer0.deserialize((JsonParser) jsonParserSequence0, (DeserializationContext) defaultDeserializationContext_Impl0);
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
      try { 
        StdDeserializer.parseDouble("h1ldP");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Double> class0 = Double.class;
      FactoryBasedEnumDeserializer factoryBasedEnumDeserializer0 = new FactoryBasedEnumDeserializer(class0, (AnnotatedMethod) null);
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object object0 = factoryBasedEnumDeserializer0._coerceNullToken(defaultDeserializationContext_Impl0, false);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PrimitiveArrayDeserializers.BooleanDeser primitiveArrayDeserializers_BooleanDeser0 = new PrimitiveArrayDeserializers.BooleanDeser();
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      primitiveArrayDeserializers_BooleanDeser0._verifyNullForPrimitive(defaultDeserializationContext_Impl0);
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, defaultDeserializationContext_Impl0);
      Class<BeanDeserializer> class0 = BeanDeserializer.class;
      ObjectReader objectReader0 = objectMapper0.readerFor(class0);
      assertNotNull(objectReader0);
  }
}