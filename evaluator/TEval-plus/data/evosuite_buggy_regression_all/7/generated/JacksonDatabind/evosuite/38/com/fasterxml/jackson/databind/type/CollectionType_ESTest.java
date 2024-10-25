/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:58:27 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.ArrayList;
import java.util.Collection;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CollectionType_ESTest extends CollectionType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Object> class0 = Object.class;
      CollectionType collectionType0 = CollectionType.construct(class0, javaType0);
      CollectionType collectionType1 = collectionType0.withContentTypeHandler(class0);
      assertFalse(collectionType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      String string0 = collectionType0.getTypeName();
      assertEquals("[collection type; class java.util.ArrayList, contains [collection type; class java.util.ArrayList, contains [simple type, class java.lang.Object]]]", string0);
      assertFalse(collectionType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_OBJECT;
      MapType mapType0 = new MapType(simpleType0, simpleType0, simpleType0);
      CollectionType collectionType0 = new CollectionType(mapType0, simpleType0);
      Class<Object> class0 = Object.class;
      JavaType javaType0 = collectionType0._narrow(class0);
      assertFalse(javaType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_OBJECT;
      MapType mapType0 = new MapType(simpleType0, simpleType0, simpleType0);
      Object object0 = new Object();
      CollectionType collectionType0 = new CollectionType(mapType0, simpleType0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      CollectionType collectionType1 = typeFactory0.constructCollectionType((Class<? extends Collection>) class0, (JavaType) collectionType0);
      CollectionType collectionType2 = collectionType1.withContentValueHandler(object0);
      assertTrue(collectionType2.equals((Object)collectionType1));
      assertFalse(collectionType2.useStaticType());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_OBJECT;
      MapType mapType0 = new MapType(simpleType0, simpleType0, simpleType0);
      CollectionType collectionType0 = new CollectionType(mapType0, simpleType0);
      CollectionType collectionType1 = collectionType0.withTypeHandler(mapType0);
      assertTrue(collectionType1.isConcrete());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_OBJECT;
      MapType mapType0 = new MapType(simpleType0, simpleType0, simpleType0);
      CollectionType collectionType0 = new CollectionType(mapType0, simpleType0);
      JavaType javaType0 = collectionType0.withContentType(mapType0);
      JavaType javaType1 = javaType0.withContentType(mapType0);
      assertSame(javaType1, javaType0);
      assertNotSame(javaType1, collectionType0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<ArrayList> class0 = ArrayList.class;
      CollectionType collectionType0 = typeFactory0.constructCollectionType(class0, class0);
      CollectionType collectionType1 = collectionType0.withStaticTyping();
      CollectionType collectionType2 = collectionType1.withStaticTyping();
      assertTrue(collectionType2.equals((Object)collectionType0));
      assertFalse(collectionType0.useStaticType());
      assertTrue(collectionType2.useStaticType());
  }
}
