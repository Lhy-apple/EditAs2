/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:15:28 GMT 2023
 */

package com.fasterxml.jackson.databind.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.annotation.NoClass;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.PlaceholderForType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.util.AccessPattern;
import com.fasterxml.jackson.databind.util.ClassUtil;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.annotation.Annotation;
import java.lang.reflect.Constructor;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.sql.BatchUpdateException;
import java.sql.ClientInfoStatus;
import java.sql.SQLClientInfoException;
import java.sql.SQLDataException;
import java.sql.SQLException;
import java.sql.SQLFeatureNotSupportedException;
import java.sql.SQLInvalidAuthorizationSpecException;
import java.sql.SQLTransientConnectionException;
import java.sql.SQLWarning;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockIOException;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.evosuite.runtime.mock.java.lang.MockError;
import org.evosuite.runtime.mock.java.lang.MockRuntimeException;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ClassUtil_ESTest extends ClassUtil_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      // Undeclared exception!
      try { 
        ClassUtil.checkAndFixAccess((Member) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      // Undeclared exception!
      try { 
        ClassUtil.getDeclaredFields((Class<?>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.util.ReflectionUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<AccessPattern> class0 = AccessPattern.class;
      Method[] methodArray0 = ClassUtil.getDeclaredMethods(class0);
      assertEquals(3, methodArray0.length);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      Class<Character> class1 = Character.class;
      List<Class<?>> list0 = ClassUtil.findRawSuperTypes(class0, class1, false);
      assertEquals(6, list0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SQLClientInfoException sQLClientInfoException0 = new SQLClientInfoException();
      Class<Object> class0 = Object.class;
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, class0, true);
      MockPrintStream mockPrintStream0 = new MockPrintStream("NULL");
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, 3, (ObjectCodec) null, mockPrintStream0);
      // Undeclared exception!
      try { 
        ClassUtil.closeOnFailAndThrowAsIOE((JsonGenerator) uTF8JsonGenerator0, (Exception) sQLClientInfoException0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.sql.SQLClientInfoException
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ClassUtil classUtil0 = new ClassUtil();
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Iterator<CollectionLikeType> iterator0 = ClassUtil.emptyIterator();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<SimpleType> class0 = SimpleType.class;
      List<Class<?>> list0 = ClassUtil.findSuperTypes(class0, class0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SQLFeatureNotSupportedException sQLFeatureNotSupportedException0 = new SQLFeatureNotSupportedException("e~cAY3dzls");
      Throwable throwable0 = ClassUtil.throwRootCauseIfIOE(sQLFeatureNotSupportedException0);
      assertEquals("java.sql.SQLFeatureNotSupportedException: e~cAY3dzls", throwable0.toString());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockIOException mockIOException0 = new MockIOException();
      // Undeclared exception!
      try { 
        ClassUtil.unwrapAndThrowAsIAE((Throwable) mockIOException0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ClassUtil.Ctor classUtil_Ctor0 = new ClassUtil.Ctor((Constructor<?>) null);
      Constructor<?> constructor0 = classUtil_Ctor0.getConstructor();
      assertNull(constructor0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ClassUtil.Ctor classUtil_Ctor0 = new ClassUtil.Ctor((Constructor<?>) null);
      // Undeclared exception!
      try { 
        classUtil_Ctor0.getDeclaringClass();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil$Ctor", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Annotation> class0 = Annotation.class;
      List<JavaType> list0 = ClassUtil.findSuperTypes((JavaType) null, class0, false);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      Class<ArrayType> class1 = ArrayType.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class1);
      List<JavaType> list0 = ClassUtil.findSuperTypes((JavaType) simpleType0, (Class<?>) class0, false);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<CollectionType> class0 = CollectionType.class;
      List<Class<?>> list0 = ClassUtil.findRawSuperTypes(class0, class0, true);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Class<Object> class1 = Object.class;
      List<Class<?>> list0 = ClassUtil.findRawSuperTypes(class1, class0, false);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<String> class0 = String.class;
      Class<Long> class1 = Long.class;
      List<Class<?>> list0 = ClassUtil.findSuperClasses(class1, class0, true);
      assertEquals(3, list0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<MapType> class0 = MapType.class;
      List<Class<?>> list0 = ClassUtil.findSuperClasses(class0, class0, false);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<SimpleType> class0 = SimpleType.class;
      Class<Annotation> class1 = Annotation.class;
      List<Class<?>> list0 = ClassUtil.findSuperClasses(class1, class0, false);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<MapType> class0 = MapType.class;
      String string0 = ClassUtil.canBeABeanType(class0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<AccessPattern> class0 = AccessPattern.class;
      String string0 = ClassUtil.canBeABeanType(class0);
      assertEquals("enum", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      String string0 = ClassUtil.canBeABeanType(class0);
      assertEquals("primitive", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<NoClass> class0 = NoClass.class;
      String string0 = ClassUtil.isLocalType(class0, true);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<Double> class0 = Double.class;
      Class<?> class1 = ClassUtil.getOuterClass(class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      boolean boolean0 = ClassUtil.isProxyType(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<Double> class0 = Double.TYPE;
      boolean boolean0 = ClassUtil.isConcrete(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<MapLikeType> class0 = MapLikeType.class;
      boolean boolean0 = ClassUtil.isConcrete(class0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<RuntimeException> class0 = RuntimeException.class;
      boolean boolean0 = ClassUtil.isCollectionMapOrArray(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<JsonMappingException> class0 = JsonMappingException.class;
      boolean boolean0 = ClassUtil.isBogusClass(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<String> class0 = String.class;
      boolean boolean0 = ClassUtil.isNonStaticInnerClass(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<Object> class0 = Object.class;
      String string0 = ClassUtil.isLocalType(class0, false);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<Double> class0 = Double.TYPE;
      ClassUtil.Ctor[] classUtil_CtorArray0 = ClassUtil.getConstructors(class0);
      assertEquals(0, classUtil_CtorArray0.length);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<Void> class0 = Void.class;
      boolean boolean0 = ClassUtil.hasClass((Object) null, class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Integer integer0 = new Integer((-3));
      Class<Character> class0 = Character.class;
      boolean boolean0 = ClassUtil.hasClass(integer0, class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      DefaultDeserializationContext defaultDeserializationContext0 = defaultDeserializationContext_Impl0.copy();
      assertNotSame(defaultDeserializationContext_Impl0, defaultDeserializationContext0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      PlaceholderForType placeholderForType0 = new PlaceholderForType(0);
      // Undeclared exception!
      try { 
        ClassUtil.verifyMustOverride(class0, placeholderForType0, (String) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Sub-class com.fasterxml.jackson.databind.type.PlaceholderForType (of class java.lang.Integer) must override method 'null'
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      MockError mockError0 = new MockError("size");
      // Undeclared exception!
      try { 
        ClassUtil.throwIfError(mockError0);
        fail("Expecting exception: Error");
      
      } catch(Error e) {
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Class<NoClass> class0 = NoClass.class;
      try { 
        ClassUtil.createInstance(class0, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Default constructor for com.fasterxml.jackson.databind.annotation.NoClass is not accessible (non-public?): not allowed to try modify access via Reflection: cannot instantiate type
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      SQLWarning sQLWarning0 = new SQLWarning("Class ");
      JsonMappingException jsonMappingException0 = JsonMappingException.from((DeserializationContext) defaultDeserializationContext_Impl0, "q", (Throwable) sQLWarning0);
      try { 
        ClassUtil.throwIfIOE(jsonMappingException0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // q
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      SQLInvalidAuthorizationSpecException sQLInvalidAuthorizationSpecException0 = new SQLInvalidAuthorizationSpecException("com.fasterxml.jackson.databind.util.ClassUtil", "Missing constructor (broken JDK (de)serialization?)");
      HashMap<String, ClientInfoStatus> hashMap0 = new HashMap<String, ClientInfoStatus>();
      SQLClientInfoException sQLClientInfoException0 = new SQLClientInfoException("/-9v*~MyWE.O", hashMap0, sQLInvalidAuthorizationSpecException0);
      Throwable throwable0 = ClassUtil.getRootCause(sQLClientInfoException0);
      assertSame(throwable0, sQLInvalidAuthorizationSpecException0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      int[] intArray0 = new int[1];
      BatchUpdateException batchUpdateException0 = new BatchUpdateException(intArray0);
      SQLDataException sQLDataException0 = new SQLDataException(batchUpdateException0);
      MockIOException mockIOException0 = new MockIOException(sQLDataException0);
      try { 
        ClassUtil.throwAsMappingException((DeserializationContext) defaultDeserializationContext_Impl0, (IOException) mockIOException0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // java.sql.SQLDataException: java.sql.BatchUpdateException
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      SimpleType simpleType0 = SimpleType.constructUnsafe(class0);
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory((DeserializerFactoryConfig) null);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.invalidTypeIdException(simpleType0, "[collection-like type; class com.fasterxml.jackson.databind.type.ArrayType, contains [simple type, class com.fasterxml.jackson.databind.type.ArrayType]]", "[collection-like type; class com.fasterxml.jackson.databind.type.ArrayType, contains [simple type, class com.fasterxml.jackson.databind.type.ArrayType]]");
      try { 
        ClassUtil.throwAsMappingException((DeserializationContext) defaultDeserializationContext_Impl0, (IOException) jsonMappingException0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Could not resolve type id '[collection-like type; class com.fasterxml.jackson.databind.type.ArrayType, contains [simple type, class com.fasterxml.jackson.databind.type.ArrayType]]' as a subtype of [simple type, class com.fasterxml.jackson.databind.type.ArrayType]: [collection-like type; class com.fasterxml.jackson.databind.type.ArrayType, contains [simple type, class com.fasterxml.jackson.databind.type.ArrayType]]
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidTypeIdException", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder(bufferRecycler0);
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      JsonGenerator jsonGenerator0 = jsonFactory0.createGenerator((OutputStream) byteArrayBuilder0, jsonEncoding0);
      SQLTransientConnectionException sQLTransientConnectionException0 = new SQLTransientConnectionException("JSON", "]GI");
      // Undeclared exception!
      try { 
        ClassUtil.closeOnFailAndThrowAsIOE(jsonGenerator0, (Closeable) jsonGenerator0, (Exception) sQLTransientConnectionException0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.sql.SQLTransientConnectionException: JSON
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      try { 
        ClassUtil.createInstance(class0, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.lang.Integer has no default (no arg) constructor
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Class<RuntimeException> class0 = RuntimeException.class;
      Constructor<RuntimeException> constructor0 = ClassUtil.findConstructor(class0, true);
      assertNotNull(constructor0);
      
      ClassUtil.Ctor classUtil_Ctor0 = new ClassUtil.Ctor(constructor0);
      classUtil_Ctor0.getParamCount();
      int int0 = classUtil_Ctor0.getParamCount();
      assertTrue(constructor0.isAccessible());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Class<String> class0 = String.class;
      Constructor<String> constructor0 = ClassUtil.findConstructor(class0, false);
      assertFalse(constructor0.isAccessible());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Short short0 = new Short((short)0);
      Class<?> class0 = ClassUtil.classOf(short0);
      assertFalse(class0.isAnnotation());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Class<?> class0 = ClassUtil.classOf((Object) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      AccessPattern accessPattern0 = AccessPattern.ALWAYS_NULL;
      AccessPattern accessPattern1 = ClassUtil.nonNull(accessPattern0, accessPattern0);
      assertSame(accessPattern1, accessPattern0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      LinkedList<Void> linkedList0 = ClassUtil.nonNull((LinkedList<Void>) null, (LinkedList<Void>) null);
      assertNull(linkedList0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      String string0 = ClassUtil.nullOrToString("`java.lang.Byte`");
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      String string0 = ClassUtil.nullOrToString((Object) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      String string0 = ClassUtil.nonNullString("Hig/d+F4F_T");
      assertEquals("Hig/d+F4F_T", string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      String string0 = ClassUtil.nonNullString((String) null);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      MockError mockError0 = new MockError("sdze");
      String string0 = ClassUtil.quotedOr(mockError0, "sdze");
      assertEquals("\"org.evosuite.runtime.mock.java.lang.MockThrowable: sdze\"", string0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      SQLException sQLException0 = new SQLException();
      MockRuntimeException mockRuntimeException0 = new MockRuntimeException(sQLException0);
      String string0 = ClassUtil.getClassDescription(mockRuntimeException0);
      assertEquals("`org.evosuite.runtime.mock.java.lang.MockRuntimeException`", string0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      String string0 = ClassUtil.classNameOf("\"B?YSRxCA`um");
      assertEquals("`java.lang.String`", string0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Class<Float> class0 = Float.TYPE;
      String string0 = ClassUtil.getClassDescription(class0);
      assertEquals("`float`", string0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      String string0 = ClassUtil.backticked((String) null);
      assertEquals("[null]", string0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Class<MapType> class0 = MapType.class;
      // Undeclared exception!
      try { 
        ClassUtil.defaultValue(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.type.MapType is not a primitive type
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Class<NoClass> class0 = NoClass.class;
      // Undeclared exception!
      try { 
        ClassUtil.wrapperType(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.annotation.NoClass is not a primitive type
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      Class<?> class1 = ClassUtil.primitiveType(class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Class<Long> class0 = Long.class;
      Class<?> class1 = ClassUtil.primitiveType(class0);
      assertNotNull(class1);
      assertEquals("long", class1.toString());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      AccessPattern accessPattern0 = AccessPattern.ALWAYS_NULL;
      Class<? extends Enum<?>> class0 = ClassUtil.findEnumType((Enum<?>) accessPattern0);
      assertEquals(16385, class0.getModifiers());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Class<AccessPattern> class0 = AccessPattern.class;
      Class<? extends Enum<?>> class1 = ClassUtil.findEnumType(class0);
      assertEquals("class com.fasterxml.jackson.databind.util.AccessPattern", class1.toString());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Class<MapLikeType> class0 = MapLikeType.class;
      Class<? extends Enum<?>> class1 = ClassUtil.findEnumType(class0);
      assertEquals("class com.fasterxml.jackson.databind.type.TypeBase", class1.toString());
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      boolean boolean0 = ClassUtil.isJacksonStdImpl((Object) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Class<String> class0 = String.class;
      boolean boolean0 = ClassUtil.isJacksonStdImpl((Object) class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Class<Short> class0 = Short.class;
      String string0 = ClassUtil.getPackageName(class0);
      assertEquals("java.lang", string0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Class<Character> class0 = Character.TYPE;
      String string0 = ClassUtil.getPackageName(class0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Class<NoClass> class0 = NoClass.class;
      Annotation[] annotationArray0 = ClassUtil.findClassAnnotations(class0);
      assertEquals(0, annotationArray0.length);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Annotation[] annotationArray0 = ClassUtil.findClassAnnotations(class0);
      assertEquals(0, annotationArray0.length);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Class<MapType> class0 = MapType.class;
      ClassUtil.Ctor[] classUtil_CtorArray0 = ClassUtil.getConstructors(class0);
      assertEquals(2, classUtil_CtorArray0.length);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      Class<?> class1 = ClassUtil.getDeclaringClass(class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Class<RuntimeException> class0 = RuntimeException.class;
      Constructor<RuntimeException> constructor0 = ClassUtil.findConstructor(class0, true);
      ClassUtil.Ctor classUtil_Ctor0 = new ClassUtil.Ctor(constructor0);
      Annotation[] annotationArray0 = classUtil_Ctor0.getDeclaredAnnotations();
      assertNotNull(annotationArray0);
      assertTrue(constructor0.isAccessible());
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Class<RuntimeException> class0 = RuntimeException.class;
      Constructor<RuntimeException> constructor0 = ClassUtil.findConstructor(class0, true);
      ClassUtil.Ctor classUtil_Ctor0 = new ClassUtil.Ctor(constructor0);
      classUtil_Ctor0.getParameterAnnotations();
      Annotation[][] annotationArray0 = classUtil_Ctor0.getParameterAnnotations();
      assertNotNull(annotationArray0);
      assertTrue(constructor0.isAccessible());
  }
}