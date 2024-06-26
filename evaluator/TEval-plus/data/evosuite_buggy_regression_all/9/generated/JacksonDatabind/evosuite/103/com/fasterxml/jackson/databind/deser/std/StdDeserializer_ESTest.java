/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:00:02 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.std;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.BuilderBasedDeserializer;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.std.DateDeserializers;
import com.fasterxml.jackson.databind.deser.std.NumberDeserializers;
import com.fasterxml.jackson.databind.deser.std.PrimitiveArrayDeserializers;
import com.fasterxml.jackson.databind.deser.std.StdDelegatingDeserializer;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.util.AccessPattern;
import com.fasterxml.jackson.databind.util.Converter;
import java.io.InputStream;
import java.io.PipedReader;
import java.io.Reader;
import java.io.StringReader;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdDeserializer_ESTest extends StdDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      NumberDeserializers.IntegerDeserializer numberDeserializers_IntegerDeserializer0 = NumberDeserializers.IntegerDeserializer.wrapperInstance;
      String string0 = numberDeserializers_IntegerDeserializer0._coercedTypeDesc();
      assertEquals("for type `java.lang.Integer`", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Converter<Module, BuilderBasedDeserializer> converter0 = (Converter<Module, BuilderBasedDeserializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<BuilderBasedDeserializer> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<BuilderBasedDeserializer>(converter0);
      StdDelegatingDeserializer<BuilderBasedDeserializer> stdDelegatingDeserializer1 = new StdDelegatingDeserializer<BuilderBasedDeserializer>(stdDelegatingDeserializer0);
      assertEquals(AccessPattern.CONSTANT, stdDelegatingDeserializer1.getNullAccessPattern());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PrimitiveArrayDeserializers.LongDeser primitiveArrayDeserializers_LongDeser0 = new PrimitiveArrayDeserializers.LongDeser();
      Converter<Object, String> converter0 = (Converter<Object, String>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<String> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<String>(converter0, (JavaType) null, primitiveArrayDeserializers_LongDeser0);
      boolean boolean0 = stdDelegatingDeserializer0._isNaN("yDe(");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Double> class0 = Double.class;
      Class<LongNode> class1 = LongNode.class;
      TypeBindings typeBindings0 = TypeBindings.create(class1, (JavaType[]) null);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      StdDelegatingDeserializer<BeanDeserializer> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<BeanDeserializer>((Converter<?, BeanDeserializer>) null);
      Converter<Object, BeanDeserializer> converter0 = (Converter<Object, BeanDeserializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      NumberDeserializers.NumberDeserializer numberDeserializers_NumberDeserializer0 = new NumberDeserializers.NumberDeserializer();
      StdDelegatingDeserializer<BeanDeserializer> stdDelegatingDeserializer1 = stdDelegatingDeserializer0.withDelegate(converter0, resolvedRecursiveType0, numberDeserializers_NumberDeserializer0);
      assertEquals(AccessPattern.CONSTANT, stdDelegatingDeserializer1.getNullAccessPattern());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DateDeserializers.SqlDateDeserializer dateDeserializers_SqlDateDeserializer0 = new DateDeserializers.SqlDateDeserializer();
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser((Reader) null);
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      // Undeclared exception!
      try { 
        dateDeserializers_SqlDateDeserializer0._parseBooleanPrimitive(jsonParser0, deserializationContext0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Converter<Object, BeanDeserializer> converter0 = (Converter<Object, BeanDeserializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<BeanDeserializer> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<BeanDeserializer>(converter0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stdDelegatingDeserializer0._parseIntPrimitive(defaultDeserializationContext_Impl0, "");
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Converter<Module, BuilderBasedDeserializer> converter0 = (Converter<Module, BuilderBasedDeserializer>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<BuilderBasedDeserializer> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<BuilderBasedDeserializer>(converter0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stdDelegatingDeserializer0._parseIntPrimitive(defaultDeserializationContext_Impl0, "d&ud#w7UB8_");
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
      NumberDeserializers.BigDecimalDeserializer numberDeserializers_BigDecimalDeserializer0 = new NumberDeserializers.BigDecimalDeserializer();
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, "not a valid float value", false);
      StringReader stringReader0 = new StringReader("not a valid float value");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        numberDeserializers_BigDecimalDeserializer0._parseLongPrimitive(readerBasedJsonParser0, defaultDeserializationContext_Impl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Converter<Double, InputStream> converter0 = (Converter<Double, InputStream>) mock(Converter.class, new ViolatedAssumptionAnswer());
      StdDelegatingDeserializer<InputStream> stdDelegatingDeserializer0 = new StdDelegatingDeserializer<InputStream>(converter0);
      JsonFactory jsonFactory0 = new JsonFactory();
      PipedReader pipedReader0 = new PipedReader();
      JsonParser jsonParser0 = jsonFactory0.createParser((Reader) pipedReader0);
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        stdDelegatingDeserializer0._parseDoublePrimitive(jsonParser0, defaultDeserializationContext_Impl0);
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
      NumberDeserializers.DoubleDeserializer numberDeserializers_DoubleDeserializer0 = NumberDeserializers.DoubleDeserializer.primitiveInstance;
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      numberDeserializers_DoubleDeserializer0.findContentNullStyle(defaultDeserializationContext_Impl0, (BeanProperty) null);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DateDeserializers.DateDeserializer dateDeserializers_DateDeserializer0 = new DateDeserializers.DateDeserializer();
      boolean boolean0 = dateDeserializers_DateDeserializer0._intOverflow(0L);
      assertFalse(boolean0);
  }
}
