/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:31:33 GMT 2023
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
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.BuilderBasedDeserializer;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.AtomicBooleanDeserializer;
import com.fasterxml.jackson.databind.deser.std.DateDeserializers;
import com.fasterxml.jackson.databind.deser.std.NullifyingDeserializer;
import com.fasterxml.jackson.databind.deser.std.NumberDeserializers;
import com.fasterxml.jackson.databind.deser.std.PrimitiveArrayDeserializers;
import com.fasterxml.jackson.databind.deser.std.StdDelegatingDeserializer;
import com.fasterxml.jackson.databind.deser.std.StringDeserializer;
import com.fasterxml.jackson.databind.deser.std.UntypedObjectDeserializer;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.util.AccessPattern;
import com.fasterxml.jackson.databind.util.Converter;
import java.io.PipedReader;
import java.io.StringReader;
import java.util.LinkedList;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdDeserializer_ESTest extends StdDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PrimitiveArrayDeserializers.BooleanDeser primitiveArrayDeserializers_BooleanDeser0 = new PrimitiveArrayDeserializers.BooleanDeser();
      String string0 = primitiveArrayDeserializers_BooleanDeser0._coercedTypeDesc();
      assertEquals("as content of type `boolean[]`", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Converter<Boolean, LinkedList<Object>> converter0 = (Converter<Boolean, LinkedList<Object>>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<LinkedList<Object>> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<LinkedList<Object>>(converter0);
      StdDelegatingDeserializer<LinkedList<Object>> stdDelegatingDeserializer1 = new StdDelegatingDeserializer<LinkedList<Object>>(stdDelegatingDeserializer0);
      assertEquals(AccessPattern.DYNAMIC, stdDelegatingDeserializer1.getEmptyAccessPattern());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      UntypedObjectDeserializer.Vanilla untypedObjectDeserializer_Vanilla0 = UntypedObjectDeserializer.Vanilla.std;
      NumberDeserializers.FloatDeserializer numberDeserializers_FloatDeserializer0 = NumberDeserializers.FloatDeserializer.primitiveInstance;
      boolean boolean0 = untypedObjectDeserializer_Vanilla0.isDefaultDeserializer(numberDeserializers_FloatDeserializer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Converter<BuilderBasedDeserializer, String> converter0 = (Converter<BuilderBasedDeserializer, String>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<String> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<String>(converter0);
      Converter<Object, String> converter1 = (Converter<Object, String>) mock(Converter.class, new ViolatedAssumptionAnswer());
      Class<String> class0 = String.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      NumberDeserializers.ShortDeserializer numberDeserializers_ShortDeserializer0 = NumberDeserializers.ShortDeserializer.wrapperInstance;
      StdDelegatingDeserializer<String> stdDelegatingDeserializer1 = stdDelegatingDeserializer0.withDelegate(converter1, simpleType0, numberDeserializers_ShortDeserializer0);
      assertEquals(AccessPattern.DYNAMIC, stdDelegatingDeserializer1.getEmptyAccessPattern());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Converter<Object, BuilderBasedDeserializer> converter0 = (Converter<Object, BuilderBasedDeserializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      NumberDeserializers.DoubleDeserializer numberDeserializers_DoubleDeserializer0 = NumberDeserializers.DoubleDeserializer.primitiveInstance;
      StdDelegatingDeserializer<BuilderBasedDeserializer> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<BuilderBasedDeserializer>(converter0, (JavaType) null, numberDeserializers_DoubleDeserializer0);
      assertEquals(AccessPattern.CONSTANT, stdDelegatingDeserializer0.getNullAccessPattern());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AtomicBooleanDeserializer atomicBooleanDeserializer0 = new AtomicBooleanDeserializer();
      JsonFactory jsonFactory0 = new JsonFactory();
      char[] charArray0 = new char[5];
      JsonParser jsonParser0 = jsonFactory0.createParser(charArray0, 753, 753);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(jsonParser0, tokenFilter0, false, false);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        atomicBooleanDeserializer0.deserialize((JsonParser) filteringParserDelegate0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Converter<Boolean, LinkedList<Object>> converter0 = (Converter<Boolean, LinkedList<Object>>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<LinkedList<Object>> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<LinkedList<Object>>(converter0);
      // Undeclared exception!
      try { 
        stdDelegatingDeserializer0._parseIntPrimitive((DeserializationContext) null, "");
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      StringDeserializer stringDeserializer0 = StringDeserializer.instance;
      // Undeclared exception!
      try { 
        stringDeserializer0._parseDoublePrimitive((DeserializationContext) null, "8,)1Ws2+/?");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate((JsonParser) null, tokenFilter0, true, true);
      DateDeserializers.CalendarDeserializer dateDeserializers_CalendarDeserializer0 = new DateDeserializers.CalendarDeserializer();
      // Undeclared exception!
      try { 
        dateDeserializers_CalendarDeserializer0.deserialize((JsonParser) filteringParserDelegate0, (DeserializationContext) defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Converter<Boolean, LinkedList<Object>> converter0 = (Converter<Boolean, LinkedList<Object>>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<LinkedList<Object>> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<LinkedList<Object>>(converter0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, stdDelegatingDeserializer0, true);
      PipedReader pipedReader0 = new PipedReader();
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory((DeserializerFactoryConfig) null);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, (DefaultSerializerProvider) null, defaultDeserializationContext_Impl0);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[7];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-32768), pipedReader0, objectMapper0, charsToNameCanonicalizer0, charArray0, 239, 73, true);
      // Undeclared exception!
      try { 
        stdDelegatingDeserializer0._parseString(readerBasedJsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Converter<BeanDeserializer, String> converter0 = (Converter<BeanDeserializer, String>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<String> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<String>(converter0);
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader(":Rp");
      JsonFactory jsonFactory0 = new JsonFactory();
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, (DefaultSerializerProvider) null, defaultDeserializationContext_Impl0);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[8];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, stringReader0, objectMapper0, charsToNameCanonicalizer0, charArray0, (-32768), 3, true);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(readerBasedJsonParser0, tokenFilter0, true, true);
      // Undeclared exception!
      try { 
        stdDelegatingDeserializer0._deserializeFromEmpty(filteringParserDelegate0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.std.StdDelegatingDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      NullifyingDeserializer nullifyingDeserializer0 = new NullifyingDeserializer();
      boolean boolean0 = nullifyingDeserializer0._isNegInf(".Bt[[");
      assertFalse(boolean0);
  }
}
