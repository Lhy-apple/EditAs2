/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:44:46 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MapType_ESTest extends MapType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      Class<String> class1 = String.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class1);
      MapType mapType1 = mapType0.withContentValueHandler(class1);
      assertTrue(mapType1.equals((Object)mapType0));
      assertFalse(mapType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      Class<String> class1 = String.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class1);
      JavaType javaType0 = mapType0._narrow(class1);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      ReferenceType referenceType0 = new ReferenceType(simpleType0, simpleType0);
      MapType mapType0 = MapType.construct((Class<?>) class0, (JavaType) referenceType0, (JavaType) simpleType0);
      MapType mapType1 = mapType0.withKeyValueHandler(simpleType0);
      assertFalse(mapType1.useStaticType());
      assertTrue(mapType1.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class0);
      MapType mapType1 = mapType0.withContentTypeHandler((Object) null);
      assertFalse(mapType1.useStaticType());
      assertTrue(mapType1.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
      String string0 = mapType0.toString();
      assertEquals("[map type; class java.lang.Object, [simple type, class java.lang.Object] -> [simple type, class java.lang.Object]]", string0);
      assertFalse(mapType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<LinkedHashSet> class0 = LinkedHashSet.class;
      CollectionType collectionType0 = typeFactory0.constructRawCollectionType(class0);
      MapType mapType0 = new MapType(collectionType0, collectionType0, collectionType0);
      assertTrue(mapType0.isContainerType());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
      MapType mapType1 = mapType0.withKeyTypeHandler(class0);
      assertFalse(mapType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<String> class0 = String.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      ReferenceType referenceType0 = new ReferenceType(simpleType0, simpleType0);
      MapType mapType0 = MapType.construct((Class<?>) class0, (JavaType) referenceType0, (JavaType) simpleType0);
      MapType mapType1 = mapType0.withValueHandler(class0);
      assertFalse(mapType1.useStaticType());
      assertTrue(mapType1.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class0);
      MapType mapType1 = mapType0.withStaticTyping();
      MapType mapType2 = mapType1.withStaticTyping();
      assertTrue(mapType2.equals((Object)mapType0));
      assertFalse(mapType0.useStaticType());
      assertTrue(mapType2.useStaticType());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
      JavaType javaType1 = mapType0.withContentType(mapType0);
      assertFalse(javaType1.useStaticType());
      assertNotSame(javaType1, mapType0);
      assertFalse(javaType1.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Object> class0 = Object.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
      JavaType javaType1 = mapType0.withContentType(javaType0);
      assertSame(javaType1, mapType0);
      assertFalse(javaType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      MapType mapType1 = mapType0.withKeyType(mapType0);
      assertNotSame(mapType1, mapType0);
      assertFalse(mapType1.equals((Object)mapType0));
      assertFalse(mapType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      MapType mapType0 = typeFactory0.constructMapType((Class<? extends Map>) class0, (JavaType) simpleType0, (JavaType) simpleType0);
      MapType mapType1 = mapType0.withKeyType(simpleType0);
      assertSame(mapType1, mapType0);
      assertFalse(mapType1.useStaticType());
  }
}