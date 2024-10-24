/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:49:21 GMT 2023
 */

package com.fasterxml.jackson.databind.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.UTF8JsonGenerator;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.annotation.NoClass;
import com.fasterxml.jackson.databind.cfg.DeserializerFactoryConfig;
import com.fasterxml.jackson.databind.deser.BeanDeserializerFactory;
import com.fasterxml.jackson.databind.deser.DefaultDeserializationContext;
import com.fasterxml.jackson.databind.node.DecimalNode;
import com.fasterxml.jackson.databind.ser.DefaultSerializerProvider;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.PlaceholderForType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.util.AccessPattern;
import com.fasterxml.jackson.databind.util.ClassUtil;
import java.io.Closeable;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.lang.reflect.AccessibleObject;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.sql.BatchUpdateException;
import java.sql.SQLDataException;
import java.sql.SQLIntegrityConstraintViolationException;
import java.sql.SQLNonTransientConnectionException;
import java.sql.SQLTransientException;
import java.sql.SQLWarning;
import java.util.Iterator;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockIOException;
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
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<NoClass> class0 = NoClass.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.weirdKeyException(class0, "hyf", "hyf");
      // Undeclared exception!
      try { 
        ClassUtil.unwrapAndThrowAsIAE((Throwable) jsonMappingException0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cannot deserialize Map key of type `com.fasterxml.jackson.databind.annotation.NoClass` from String \"hyf\": hyf
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Double> class0 = Double.class;
      Type[] typeArray0 = ClassUtil.getGenericInterfaces(class0);
      assertEquals(1, typeArray0.length);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Field[] fieldArray0 = ClassUtil.getDeclaredFields(class0);
      assertEquals(0, fieldArray0.length);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      Method[] methodArray0 = ClassUtil.getClassMethods(class0);
      assertEquals(34, methodArray0.length);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      Class<IOException> class1 = IOException.class;
      List<Class<?>> list0 = ClassUtil.findSuperTypes(class0, class1);
      assertEquals(7, list0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SQLTransientException sQLTransientException0 = new SQLTransientException("9CmtnC@{x/R'O,", "9CmtnC@{x/R'O,", 3346);
      // Undeclared exception!
      try { 
        ClassUtil.closeOnFailAndThrowAsIOE((JsonGenerator) null, (Exception) sQLTransientException0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ClassUtil classUtil0 = new ClassUtil();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Iterator<DecimalNode> iterator0 = ClassUtil.emptyIterator();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<AccessibleObject> class0 = AccessibleObject.class;
      try { 
        ClassUtil.findConstructor(class0, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Default constructor for java.lang.reflect.AccessibleObject is not accessible (non-public?): not allowed to try modify access via Reflection: cannot instantiate type
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Void> class0 = Void.class;
      Class class1 = (Class)ClassUtil.getGenericSuperclass(class0);
      assertFalse(class1.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<RuntimeException> class0 = RuntimeException.class;
      Constructor<RuntimeException> constructor0 = ClassUtil.findConstructor(class0, false);
      ClassUtil.Ctor classUtil_Ctor0 = new ClassUtil.Ctor(constructor0);
      Constructor<?> constructor1 = classUtil_Ctor0.getConstructor();
      assertNotNull(constructor1);
      assertFalse(constructor1.isAccessible());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<NoClass> class0 = NoClass.class;
      List<JavaType> list0 = ClassUtil.findSuperTypes((JavaType) null, class0, false);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Object> class0 = Object.class;
      List<Class<?>> list0 = ClassUtil.findRawSuperTypes(class0, class0, false);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<AccessPattern> class0 = AccessPattern.class;
      Class<MapLikeType> class1 = MapLikeType.class;
      List<Class<?>> list0 = ClassUtil.findSuperClasses(class1, class0, true);
      assertEquals(5, list0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<MapLikeType> class0 = MapLikeType.class;
      List<Class<?>> list0 = ClassUtil.findSuperClasses(class0, class0, false);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<SimpleType> class0 = SimpleType.class;
      Class<? extends Enum<?>> class1 = ClassUtil.findEnumType(class0);
      Class<JsonMappingException> class2 = JsonMappingException.class;
      List<Class<?>> list0 = ClassUtil.findSuperClasses(class1, class2, false);
      assertEquals(3, list0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      String string0 = ClassUtil.canBeABeanType(class0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<IOException> class0 = IOException.class;
      String string0 = ClassUtil.isLocalType(class0, true);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<AccessPattern> class0 = AccessPattern.class;
      String string0 = ClassUtil.isLocalType(class0, false);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<RuntimeException> class0 = RuntimeException.class;
      Class<?> class1 = ClassUtil.getOuterClass(class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<AccessPattern> class0 = AccessPattern.class;
      boolean boolean0 = ClassUtil.isProxyType(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      boolean boolean0 = ClassUtil.isConcrete(class0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<RuntimeException> class0 = RuntimeException.class;
      boolean boolean0 = ClassUtil.isCollectionMapOrArray(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<RuntimeException> class0 = RuntimeException.class;
      boolean boolean0 = ClassUtil.isBogusClass(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<PlaceholderForType> class0 = PlaceholderForType.class;
      boolean boolean0 = ClassUtil.isNonStaticInnerClass(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<Object> class0 = Object.class;
      boolean boolean0 = ClassUtil.hasEnclosingMethod(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<Short> class0 = Short.TYPE;
      Annotation[] annotationArray0 = ClassUtil.findClassAnnotations(class0);
      assertEquals(0, annotationArray0.length);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      boolean boolean0 = ClassUtil.hasClass((Object) null, class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      Class<CollectionType> class1 = CollectionType.class;
      boolean boolean0 = ClassUtil.hasClass(class0, class1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      BeanDeserializerFactory beanDeserializerFactory0 = BeanDeserializerFactory.instance;
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      DefaultDeserializationContext defaultDeserializationContext0 = defaultDeserializationContext_Impl0.copy();
      assertEquals(0, defaultDeserializationContext0.getDeserializationFeatures());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<AccessPattern> class0 = AccessPattern.class;
      Short short0 = new Short((short) (-2714));
      // Undeclared exception!
      try { 
        ClassUtil.verifyMustOverride(class0, short0, "entrySet");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Sub-class java.lang.Short (of class com.fasterxml.jackson.databind.util.AccessPattern) must override method 'entrySet'
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      MockError mockError0 = new MockError();
      SQLDataException sQLDataException0 = new SQLDataException("seriaVersnUID", mockError0);
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLDataException0, (Object) mockError0, (-857));
      // Undeclared exception!
      try { 
        ClassUtil.unwrapAndThrowAsIAE((Throwable) jsonMappingException0);
        fail("Expecting exception: Error");
      
      } catch(Error e) {
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      MockError mockError0 = new MockError("<D7rq(;");
      Throwable throwable0 = ClassUtil.throwIfIOE(mockError0);
      assertSame(throwable0, mockError0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Class<PlaceholderForType> class0 = PlaceholderForType.class;
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      ObjectIdGenerators.IntSequenceGenerator objectIdGenerators_IntSequenceGenerator0 = new ObjectIdGenerators.IntSequenceGenerator();
      IOContext iOContext0 = new IOContext(bufferRecycler0, objectIdGenerators_IntSequenceGenerator0, true);
      JsonFactory jsonFactory0 = new JsonFactory();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("JSON", false);
      UTF8JsonGenerator uTF8JsonGenerator0 = new UTF8JsonGenerator(iOContext0, (-2429), objectMapper0, mockFileOutputStream0);
      SQLIntegrityConstraintViolationException sQLIntegrityConstraintViolationException0 = new SQLIntegrityConstraintViolationException("JSON", "JSON");
      SQLWarning sQLWarning0 = new SQLWarning("com.fasterxml.jackson.databind.type", sQLIntegrityConstraintViolationException0);
      MockRuntimeException mockRuntimeException0 = new MockRuntimeException("com.fasterxml.jackson.databind.type", sQLWarning0);
      SQLNonTransientConnectionException sQLNonTransientConnectionException0 = new SQLNonTransientConnectionException("com.fasterxml.jackson.databind.type", mockRuntimeException0);
      JsonMappingException.Reference jsonMappingException_Reference0 = new JsonMappingException.Reference(class0);
      JsonMappingException jsonMappingException0 = JsonMappingException.wrapWithPath((Throwable) sQLNonTransientConnectionException0, jsonMappingException_Reference0);
      try { 
        ClassUtil.closeOnFailAndThrowAsIOE((JsonGenerator) uTF8JsonGenerator0, (Closeable) uTF8JsonGenerator0, (Exception) jsonMappingException0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // com.fasterxml.jackson.databind.type (through reference chain: com.fasterxml.jackson.databind.type.PlaceholderForType[?])
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      JsonFactory jsonFactory0 = new JsonFactory();
      DefaultSerializerProvider.Impl defaultSerializerProvider_Impl0 = new DefaultSerializerProvider.Impl();
      ObjectMapper objectMapper0 = new ObjectMapper(jsonFactory0, defaultSerializerProvider_Impl0, (DefaultDeserializationContext) null);
      DeserializationContext deserializationContext0 = objectMapper0.getDeserializationContext();
      int[] intArray0 = new int[6];
      BatchUpdateException batchUpdateException0 = new BatchUpdateException("JSON", intArray0);
      MockIOException mockIOException0 = new MockIOException("JSON", batchUpdateException0);
      try { 
        ClassUtil.throwAsMappingException(deserializationContext0, (IOException) mockIOException0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // JSON
         //
         verifyException("com.fasterxml.jackson.databind.JsonMappingException", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      DeserializerFactoryConfig deserializerFactoryConfig0 = new DeserializerFactoryConfig();
      BeanDeserializerFactory beanDeserializerFactory0 = new BeanDeserializerFactory(deserializerFactoryConfig0);
      DefaultDeserializationContext.Impl defaultDeserializationContext_Impl0 = new DefaultDeserializationContext.Impl(beanDeserializerFactory0);
      Class<IOException> class0 = IOException.class;
      JsonMappingException jsonMappingException0 = defaultDeserializationContext_Impl0.weirdKeyException(class0, "<D7rq(;", "hyf");
      try { 
        ClassUtil.throwAsMappingException((DeserializationContext) defaultDeserializationContext_Impl0, (IOException) jsonMappingException0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Cannot deserialize Map key of type `java.io.IOException` from String \"<D7rq(;\": hyf
         //
         verifyException("com.fasterxml.jackson.databind.exc.InvalidFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Object object0 = ClassUtil.createInstance(class0, false);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Class<MapLikeType> class0 = MapLikeType.class;
      try { 
        ClassUtil.createInstance(class0, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.type.MapLikeType has no default (no arg) constructor
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Constructor<Object> constructor0 = ClassUtil.findConstructor(class0, true);
      assertTrue(constructor0.isAccessible());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Class<Short> class0 = Short.class;
      Class<?> class1 = ClassUtil.classOf(class0);
      assertFalse(class1.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Class<?> class0 = ClassUtil.classOf((Object) null);
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Long long0 = new Long(0L);
      Long long1 = ClassUtil.nonNull(long0, long0);
      assertEquals(0L, (long)long1);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Short short0 = new Short((short) (-2714));
      Short short1 = ClassUtil.nonNull((Short) null, short0);
      assertEquals((short) (-2714), (short)short1);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      String string0 = ClassUtil.nullOrToString("unknown");
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      String string0 = ClassUtil.nonNullString("9CmtnC@{x/R'O,");
      assertEquals("9CmtnC@{x/R'O,", string0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      String string0 = ClassUtil.quotedOr(simpleObjectIdResolver0, "serialVersionUID");
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      String string0 = ClassUtil.quotedOr((Object) null, "`java.sql.SQLNonTransientConnectionException`");
      assertEquals("`java.sql.SQLNonTransientConnectionException`", string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      String string0 = ClassUtil.getClassDescription((Object) null);
      assertEquals("unknown", string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      String string0 = ClassUtil.getClassDescription("ZVtt\"Es/1");
      assertEquals("`java.lang.String`", string0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      SQLNonTransientConnectionException sQLNonTransientConnectionException0 = new SQLNonTransientConnectionException((String) null, "Sub-class %s (of class %s) must override method '%s'", 0);
      String string0 = ClassUtil.classNameOf(sQLNonTransientConnectionException0);
      assertEquals("`java.sql.SQLNonTransientConnectionException`", string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      String string0 = ClassUtil.classNameOf((Object) null);
      assertEquals("[null]", string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Class<Short> class0 = Short.TYPE;
      String string0 = ClassUtil.getClassDescription(class0);
      assertEquals("`short`", string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      String string0 = ClassUtil.backticked((String) null);
      assertEquals("[null]", string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Class<RuntimeException> class0 = RuntimeException.class;
      // Undeclared exception!
      try { 
        ClassUtil.defaultValue(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class java.lang.RuntimeException is not a primitive type
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Class<ArrayType> class0 = ArrayType.class;
      // Undeclared exception!
      try { 
        ClassUtil.wrapperType(class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Class com.fasterxml.jackson.databind.type.ArrayType is not a primitive type
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      Class<?> class1 = ClassUtil.primitiveType(class0);
      assertNull(class1);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      boolean boolean0 = ClassUtil.isJacksonStdImpl((Object) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      boolean boolean0 = ClassUtil.isJacksonStdImpl((Object) "com.fasterxml.jackson.databind.type");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      String string0 = ClassUtil.getPackageName(class0);
      assertEquals("com.fasterxml.jackson.databind.type", string0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Class<Short> class0 = Short.class;
      Annotation[] annotationArray0 = ClassUtil.findClassAnnotations(class0);
      assertEquals(0, annotationArray0.length);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      ClassUtil.Ctor[] classUtil_CtorArray0 = ClassUtil.getConstructors(class0);
      assertEquals(2, classUtil_CtorArray0.length);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Class<SimpleType> class0 = SimpleType.class;
      ClassUtil.getDeclaringClass(class0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ClassUtil.getEnclosingClass(class0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Class<RuntimeException> class0 = RuntimeException.class;
      Constructor<RuntimeException> constructor0 = ClassUtil.findConstructor(class0, false);
      ClassUtil.Ctor classUtil_Ctor0 = new ClassUtil.Ctor(constructor0);
      classUtil_Ctor0.getParamCount();
      classUtil_Ctor0.getParamCount();
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      ClassUtil.Ctor classUtil_Ctor0 = new ClassUtil.Ctor((Constructor<?>) null);
      // Undeclared exception!
      try { 
        classUtil_Ctor0.getDeclaredAnnotations();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil$Ctor", e);
      }
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      ClassUtil.Ctor classUtil_Ctor0 = new ClassUtil.Ctor((Constructor<?>) null);
      // Undeclared exception!
      try { 
        classUtil_Ctor0.getParameterAnnotations();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.util.ClassUtil$Ctor", e);
      }
  }
}
