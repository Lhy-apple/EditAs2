/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:05:14 GMT 2023
 */

package com.fasterxml.jackson.databind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.InjectableValues;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.deser.DeserializerFactory;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.PlaceholderForType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.util.ObjectBuffer;
import java.io.IOException;
import java.io.PipedReader;
import java.io.PipedWriter;
import java.sql.SQLTransactionRollbackException;
import java.text.DateFormat;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.text.MockDateFormat;
import org.evosuite.runtime.mock.java.text.MockSimpleDateFormat;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DeserializationContext_ESTest extends DeserializationContext_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      PlaceholderForType placeholderForType0 = new PlaceholderForType(0);
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) placeholderForType0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = (DefaultDeserializationContext.Impl)objectReader0._context;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.parseDate("");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      Class<MockDateFormat> class0 = MockDateFormat.class;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      defaultDeserializationContext_Impl0.weirdKeyException(class0, "D", "");
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Integer integer0 = new Integer(2);
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      defaultDeserializationContext_Impl0.weirdNumberException(integer0, class0, "");
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      DefaultDeserializationContext defaultDeserializationContext0 = defaultDeserializationContext_Impl0.copy();
      assertEquals(0, defaultDeserializationContext0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.reportBadMerge((JsonDeserializer<?>) null);
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
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectMapper objectMapper0 = new ObjectMapper();
      JsonFactory jsonFactory0 = new JsonFactory(objectMapper0);
      BufferRecycler bufferRecycler0 = jsonFactory0._getBufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, "", true);
      PipedWriter pipedWriter0 = new PipedWriter();
      PipedReader pipedReader0 = new PipedReader(pipedWriter0);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[5];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 762, pipedReader0, objectMapper0, charsToNameCanonicalizer0, charArray0, 693, (-2195), true);
      JsonToken jsonToken0 = JsonToken.END_OBJECT;
      Object[] objectArray0 = new Object[0];
      try { 
        defaultDeserializationContext_Impl0.reportWrongTokenException((JsonParser) readerBasedJsonParser0, jsonToken0, "", objectArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unexpected token (null), expected END_OBJECT: 
         //  at [Source: UNKNOWN; line: 1, column: 0]
         //
         verifyException("com.fasterxml.jackson.databind.exc.MismatchedInputException", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      MockDate mockDate0 = new MockDate();
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.constructCalendar(mockDate0);
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
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<DateFormat> class0 = DateFormat.class;
      Class<Object> class1 = Object.class;
      JavaType[] javaTypeArray0 = new JavaType[0];
      TypeBindings typeBindings0 = TypeBindings.create(class1, javaTypeArray0);
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      try { 
        defaultDeserializationContext_Impl0.reportInputMismatch((JavaType) resolvedRecursiveType0, "JSON", (Object[]) javaTypeArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // JSON
         //
         verifyException("com.fasterxml.jackson.databind.exc.MismatchedInputException", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<PlaceholderForType> class0 = PlaceholderForType.class;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.mappingException(class0);
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
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.canOverrideAccessModifiers();
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
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Throwable> class0 = Throwable.class;
      Object[] objectArray0 = new Object[4];
      try { 
        defaultDeserializationContext_Impl0.reportInputMismatch((Class<?>) class0, "B|SWC$0", objectArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // B|SWC$0
         //
         verifyException("com.fasterxml.jackson.databind.exc.MismatchedInputException", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JsonToken jsonToken0 = JsonToken.NOT_AVAILABLE;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.reportWrongTokenException((JavaType) resolvedRecursiveType0, jsonToken0, "=,/@", (Object[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DatabindContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object[] objectArray0 = new Object[2];
      defaultDeserializationContext_Impl0.mappingException("D", objectArray0);
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.getDefaultPropertyFormat(class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.findClass("|)MSTv]M?p5J{@Yk]");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object[] objectArray0 = new Object[9];
      try { 
        defaultDeserializationContext_Impl0.reportMappingException("2*cG:", objectArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // 2*cG:
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonToken jsonToken0 = JsonToken.VALUE_EMBEDDED_OBJECT;
      Object[] objectArray0 = new Object[6];
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.reportWrongTokenException((JsonDeserializer<?>) null, jsonToken0, "Cannot add mapping from class ", objectArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      defaultDeserializationContext_Impl0.mappingException("S-@rY*Zuj");
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      Class<Integer> class0 = Integer.class;
      InjectableValues.Std injectableValues_Std0 = new InjectableValues.Std();
      ObjectReader objectReader0 = objectMapper0.reader((InjectableValues) injectableValues_Std0);
      TypeFactory typeFactory0 = objectReader0.getTypeFactory();
      MapLikeType mapLikeType0 = typeFactory0.constructRawMapLikeType(class0);
      try { 
        deserializationContext0.resolveSubType(mapLikeType0, "#cZLgAH");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Could not resolve type id '#cZLgAH' as a subtype of [map-like type; class java.lang.Integer, [simple type, class java.lang.Object] -> [simple type, class java.lang.Object]]: problem: (java.lang.NullPointerException) null
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidTypeIdException", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.readValue((JsonParser) null, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.getBase64Variant();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<Integer> class0 = Integer.class;
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      JsonParser jsonParser0 = arrayNode0.traverse();
      JsonToken jsonToken0 = JsonToken.START_ARRAY;
      try { 
        defaultDeserializationContext_Impl0.reportTrailingTokens((Class<?>) class0, jsonParser0, jsonToken0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Trailing token (of type START_ARRAY) found after value (bound as `java.lang.Integer`): not allowed as per `DeserializationFeature.FAIL_ON_TRAILING_TOKENS`
         //  at [Source: UNKNOWN; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.exc.MismatchedInputException", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationFeature deserializationFeature0 = DeserializationFeature.READ_UNKNOWN_ENUM_VALUES_USING_DEFAULT_VALUE;
      DeserializationFeature[] deserializationFeatureArray0 = new DeserializationFeature[3];
      deserializationFeatureArray0[0] = deserializationFeature0;
      deserializationFeatureArray0[1] = deserializationFeature0;
      deserializationFeatureArray0[2] = deserializationFeature0;
      ObjectReader objectReader0 = objectMapper0.reader(deserializationFeature0, deserializationFeatureArray0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = (DefaultDeserializationContext.Impl)objectReader0._context;
      Class<Object> class0 = Object.class;
      defaultDeserializationContext_Impl0.weirdStringException("(/,", class0, "DeserializationProblemHandler.handleWeirdNativeValue() for type %s returned value of type %s");
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      DeserializationFeature deserializationFeature0 = DeserializationFeature.READ_UNKNOWN_ENUM_VALUES_USING_DEFAULT_VALUE;
      DeserializationFeature[] deserializationFeatureArray0 = new DeserializationFeature[3];
      deserializationFeatureArray0[0] = deserializationFeature0;
      deserializationFeatureArray0[1] = deserializationFeature0;
      deserializationFeatureArray0[2] = deserializationFeature0;
      ObjectReader objectReader0 = objectMapper0.reader(deserializationFeature0, deserializationFeatureArray0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = (DefaultDeserializationContext.Impl)objectReader0._context;
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.instantiationException(class0, "(/,");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      Class<CollectionType> class0 = CollectionType.class;
      PlaceholderForType placeholderForType0 = new PlaceholderForType(0);
      MapLikeType mapLikeType0 = MapLikeType.upgradeFrom(placeholderForType0, placeholderForType0, placeholderForType0);
      JavaType[] javaTypeArray0 = new JavaType[11];
      javaTypeArray0[7] = (JavaType) placeholderForType0;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (TypeBindings) null, (JavaType) mapLikeType0, javaTypeArray0, javaTypeArray0[7]);
      Class<MockSimpleDateFormat> class1 = MockSimpleDateFormat.class;
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) referenceType0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = (DefaultDeserializationContext.Impl)objectReader0._context;
      JsonToken jsonToken0 = JsonToken.END_ARRAY;
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.reportWrongTokenException((Class<?>) class1, jsonToken0, "", (Object[]) javaTypeArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Class<Object> class0 = Object.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      Class<PlaceholderForType> class1 = PlaceholderForType.class;
      defaultDeserializationContext_Impl0.weirdNativeValueException(resolvedRecursiveType0, class1);
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Object[] objectArray0 = new Object[5];
      try { 
        defaultDeserializationContext_Impl0.reportMissingContent("AirgLo,8rK|x.D", objectArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No content to map due to end-of-input
         //
         verifyException("com.fasterxml.jackson.databind.exc.MismatchedInputException", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      defaultDeserializationContext_Impl0.getActiveView();
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<String> class0 = String.class;
      JsonToken jsonToken0 = JsonToken.VALUE_EMBEDDED_OBJECT;
      defaultDeserializationContext_Impl0.mappingException(class0, jsonToken0);
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Integer integer0 = new Integer(3523);
      SQLTransactionRollbackException sQLTransactionRollbackException0 = new SQLTransactionRollbackException("]yv@awuSH|+tCjJ9", " bytes");
      // Undeclared exception!
      try { 
        defaultDeserializationContext_Impl0.setAttribute(integer0, sQLTransactionRollbackException0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = null;
      try {
        defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl((DeserializerFactory) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cannot pass null DeserializerFactory
         //
         verifyException("com.fasterxml.jackson.databind.DeserializationContext", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      defaultDeserializationContext_Impl0.getContextualType();
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonDeserializer<CollectionType> jsonDeserializer0 = (JsonDeserializer<CollectionType>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      JsonDeserializer<BeanDeserializer> jsonDeserializer1 = (JsonDeserializer<BeanDeserializer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      defaultDeserializationContext_Impl0.reportUnknownProperty(jsonDeserializer0, "Cannot construct instance of %s: %s", jsonDeserializer1);
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      boolean boolean0 = defaultDeserializationContext_Impl0.hasDeserializationFeatures(1335);
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      boolean boolean0 = defaultDeserializationContext_Impl0.hasDeserializationFeatures(0);
      assertTrue(boolean0);
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      boolean boolean0 = defaultDeserializationContext_Impl0.hasSomeOfFeatures(3);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectBuffer objectBuffer0 = defaultDeserializationContext_Impl0.leaseObjectBuffer();
      assertNotNull(objectBuffer0);
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      ObjectBuffer objectBuffer0 = new ObjectBuffer();
      defaultDeserializationContext_Impl0.returnObjectBuffer(objectBuffer0);
      assertEquals(0, defaultDeserializationContext_Impl0.getDeserializationFeatures());
  }
}