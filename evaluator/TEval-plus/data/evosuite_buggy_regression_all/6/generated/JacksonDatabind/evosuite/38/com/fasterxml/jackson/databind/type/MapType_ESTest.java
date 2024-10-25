/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:22:58 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBase;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.time.format.ResolverStyle;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MapType_ESTest extends MapType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JavaType javaType0 = TypeBase._bogusSuperClass(class0);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(javaType0, javaType0);
      MapType mapType0 = new MapType(referenceType0, referenceType0, referenceType0);
      MapType mapType1 = MapType.construct((Class<?>) class0, (JavaType) mapType0, (JavaType) referenceType0);
      MapType mapType2 = mapType1.withKeyValueHandler(referenceType0);
      assertFalse(mapType2.useStaticType());
      assertTrue(mapType2.equals((Object)mapType1));
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JavaType javaType0 = TypeBase._bogusSuperClass(class0);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(javaType0, javaType0);
      MapType mapType0 = new MapType(referenceType0, referenceType0, referenceType0);
      Object object0 = new Object();
      MapType mapType1 = mapType0.withContentValueHandler(object0);
      assertNotSame(mapType0, mapType1);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JavaType javaType0 = TypeBase._bogusSuperClass(class0);
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
      Class<Object> class1 = Object.class;
      JavaType javaType1 = mapType0._narrow(class1);
      assertFalse(javaType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JavaType javaType0 = TypeBase._bogusSuperClass(class0);
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
      Integer integer0 = new Integer(1);
      MapType mapType1 = mapType0.withKeyValueHandler(integer0);
      assertFalse(mapType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JavaType javaType0 = TypeBase._bogusSuperClass(class0);
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
      MapType mapType1 = mapType0.withContentTypeHandler(mapType0);
      assertFalse(mapType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      MapLikeType mapLikeType0 = new MapLikeType(simpleType0, simpleType0, simpleType0);
      MapType mapType0 = new MapType(mapLikeType0, mapLikeType0, mapLikeType0);
      String string0 = mapType0.toString();
      assertEquals("[map type; class long, [map-like type; class long, [simple type, class long] -> [simple type, class long]] -> [map-like type; class long, [simple type, class long] -> [simple type, class long]]]", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Class<ResolverStyle> class1 = ResolverStyle.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class1);
      MapType mapType1 = mapType0.withKeyTypeHandler(typeFactory0);
      assertFalse(mapType1.useStaticType());
      assertTrue(mapType1.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Class<ResolverStyle> class1 = ResolverStyle.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class0, class1);
      MapType mapType1 = mapType0.withStaticTyping();
      MapType mapType2 = mapType1.withStaticTyping();
      assertNotSame(mapType2, mapType0);
      assertTrue(mapType2.useStaticType());
      assertTrue(mapType2.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JavaType javaType0 = TypeBase._bogusSuperClass(class0);
      ReferenceType referenceType0 = ReferenceType.upgradeFrom(javaType0, javaType0);
      MapType mapType0 = new MapType(referenceType0, referenceType0, referenceType0);
      JavaType javaType1 = mapType0.withContentType(mapType0);
      assertNotSame(javaType1, mapType0);
      assertFalse(javaType1.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_LONG;
      MapLikeType mapLikeType0 = new MapLikeType(simpleType0, simpleType0, simpleType0);
      MapType mapType0 = new MapType(mapLikeType0, mapLikeType0, mapLikeType0);
      JavaType javaType0 = mapType0.withContentType(mapLikeType0);
      assertSame(javaType0, mapType0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class0 = HashMap.class;
      Class<Object> class1 = Object.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class1, class1);
      MapType mapType1 = mapType0.withKeyType(mapType0);
      assertNotSame(mapType1, mapType0);
      assertFalse(mapType1.equals((Object)mapType0));
      assertFalse(mapType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      JavaType javaType0 = TypeBase._bogusSuperClass(class0);
      MapType mapType0 = MapType.construct(class0, javaType0, javaType0);
      MapType mapType1 = mapType0.withKeyType(javaType0);
      assertFalse(mapType1.useStaticType());
      assertSame(mapType1, mapType0);
  }
}
