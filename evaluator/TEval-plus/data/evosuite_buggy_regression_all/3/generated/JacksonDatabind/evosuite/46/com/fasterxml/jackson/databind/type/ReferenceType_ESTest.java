/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:39:12 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.fasterxml.jackson.databind.type.TypeParser;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ReferenceType_ESTest extends ReferenceType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      String string0 = referenceType0.getTypeName();
      assertEquals("[reference type, class java.lang.Object<int<[simple type, class int]>]", string0);
      assertFalse(referenceType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      Object object0 = new Object();
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, (Object) class0, object0);
      TypeFactory typeFactory0 = TypeFactory.instance;
      ArrayType arrayType0 = typeFactory0.constructArrayType((JavaType) referenceType0);
      StringBuilder stringBuilder0 = new StringBuilder();
      arrayType0.getGenericSignature(stringBuilder0);
      assertEquals("[Ljava/lang/String<Z;;", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<ReferenceType> class0 = ReferenceType.class;
      Class<Object> class1 = Object.class;
      Class<Integer> class2 = Integer.class;
      SimpleType simpleType0 = new SimpleType(class2);
      MapLikeType mapLikeType0 = MapLikeType.construct(class1, simpleType0, simpleType0);
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) mapLikeType0, (Object) class1, (Object) simpleType0);
      referenceType0.getReferencedType();
      assertFalse(referenceType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      TypeParser typeParser0 = new TypeParser(typeFactory0);
      JavaType javaType0 = typeFactory0._unknownType();
      Object object0 = new Object();
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, javaType0, (Object) typeParser0, object0);
      boolean boolean0 = referenceType0.isReferenceType();
      assertFalse(referenceType0.useStaticType());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      String string0 = referenceType0.getErasedSignature();
      assertFalse(referenceType0.useStaticType());
      assertEquals("Ljava/lang/Object;", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
      Integer integer0 = new Integer((-1382));
      ReferenceType referenceType0 = new ReferenceType(class0, mapType0, integer0, mapType0, true);
      Class<?> class1 = referenceType0.getParameterSource();
      assertFalse(class1.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      JavaType javaType0 = referenceType0._narrow(class1);
      assertFalse(javaType0.isJavaLangObject());
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = new SimpleType(class0);
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      Object object0 = new Object();
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) mapLikeType0, object0, (Object) mapLikeType0);
      int int0 = referenceType0.containedTypeCount();
      assertEquals(1, int0);
      assertFalse(referenceType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      ReferenceType referenceType1 = referenceType0.withTypeHandler(referenceType0);
      assertNotSame(referenceType1, referenceType0);
      assertFalse(referenceType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      ReferenceType referenceType1 = referenceType0.withTypeHandler((Object) null);
      assertFalse(referenceType1.useStaticType());
      assertSame(referenceType1, referenceType0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      ReferenceType referenceType1 = referenceType0.withContentTypeHandler(object0);
      boolean boolean0 = referenceType1.equals(referenceType0);
      assertTrue(boolean0);
      assertFalse(referenceType1.useStaticType());
      assertNotSame(referenceType1, referenceType0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      ReferenceType referenceType1 = referenceType0.withContentTypeHandler((Object) null);
      assertFalse(referenceType1.useStaticType());
      assertSame(referenceType1, referenceType0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      ReferenceType referenceType1 = ReferenceType.construct((Class<?>) class0, (JavaType) referenceType0, object0, (Object) jsonEncoding0);
      ReferenceType referenceType2 = referenceType1.withContentValueHandler(referenceType0);
      assertNotSame(referenceType2, referenceType1);
      assertFalse(referenceType2.useStaticType());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      Object object0 = new Object();
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, (Object) class0, object0);
      ReferenceType referenceType1 = referenceType0.withValueHandler((Object) null);
      assertFalse(referenceType1.useStaticType());
      assertSame(referenceType1, referenceType0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      Object object0 = new Object();
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, (Object) class0, object0);
      ReferenceType referenceType1 = referenceType0.withContentValueHandler((Object) null);
      assertFalse(referenceType1.useStaticType());
      assertSame(referenceType1, referenceType0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      ReferenceType referenceType1 = referenceType0.withStaticTyping();
      assertFalse(referenceType0.useStaticType());
      assertTrue(referenceType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Object object0 = new Object();
      ReferenceType referenceType0 = new ReferenceType(class0, simpleType0, object0, object0, true);
      ReferenceType referenceType1 = referenceType0.withStaticTyping();
      assertSame(referenceType1, referenceType0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      JavaType javaType0 = referenceType0.containedType(117);
      assertFalse(referenceType0.useStaticType());
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      JavaType javaType0 = referenceType0.containedType((byte)0);
      assertFalse(referenceType0.useStaticType());
      assertNotNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      Object object0 = new Object();
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, (Object) class0, object0);
      String string0 = referenceType0.containedTypeName((-808));
      assertFalse(referenceType0.useStaticType());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      CollectionType collectionType0 = CollectionType.construct(class0, simpleType0);
      Object object0 = new Object();
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) collectionType0, (Object) simpleType0, object0);
      String string0 = referenceType0.containedTypeName(0);
      assertEquals("T", string0);
      assertNotNull(string0);
      assertFalse(referenceType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<Object> class0 = Object.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Object object0 = new Object();
      ReferenceType referenceType0 = new ReferenceType(class0, simpleType0, object0, object0, true);
      boolean boolean0 = referenceType0.equals(class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      Object object0 = new Object();
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, (Object) class0, object0);
      boolean boolean0 = referenceType0.equals(referenceType0);
      assertFalse(referenceType0.useStaticType());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = new SimpleType(class0);
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      Object object0 = new Object();
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) mapLikeType0, object0, (Object) mapLikeType0);
      boolean boolean0 = referenceType0.equals((Object) null);
      assertFalse(referenceType0.useStaticType());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Class<Integer> class1 = Integer.TYPE;
      SimpleType simpleType0 = SimpleType.construct(class1);
      Object object0 = new Object();
      JsonEncoding jsonEncoding0 = JsonEncoding.UTF32_LE;
      ReferenceType referenceType0 = ReferenceType.construct((Class<?>) class0, (JavaType) simpleType0, object0, (Object) jsonEncoding0);
      Class<String> class2 = String.class;
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory(true);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      ReferenceType referenceType1 = ReferenceType.construct((Class<?>) class2, (JavaType) referenceType0, (Object) objectNode0, (Object) objectNode0);
      boolean boolean0 = referenceType1.equals(referenceType0);
      assertFalse(boolean0);
      assertFalse(referenceType1.useStaticType());
  }
}