/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:41:18 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotatedWithParams;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ExternalTypeHandler_ESTest extends ExternalTypeHandler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder();
      Class<Object> class0 = Object.class;
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType javaType0 = TypeFactory.unknownType();
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(javaType0, javaType0);
      ArrayType arrayType0 = ArrayType.construct((JavaType) collectionLikeType0, typeBindings0, (Object) class0, (Object) class0);
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.EXTERNAL_PROPERTY;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(collectionLikeType0, (TypeIdResolver) null, "", true, collectionLikeType0, jsonTypeInfo_As0);
      AnnotationMap annotationMap0 = new AnnotationMap();
      AnnotatedParameter annotatedParameter0 = new AnnotatedParameter((AnnotatedWithParams) null, javaType0, annotationMap0, 2578);
      Integer integer0 = new Integer(2578);
      PropertyMetadata propertyMetadata0 = PropertyMetadata.STD_OPTIONAL;
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, arrayType0, propertyName0, asPropertyTypeDeserializer0, annotationMap0, annotatedParameter0, 2578, integer0, propertyMetadata0);
      externalTypeHandler_Builder0.addExternal(creatorProperty0, asPropertyTypeDeserializer0);
      assertFalse(creatorProperty0.isVirtual());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ExternalTypeHandler externalTypeHandler0 = null;
      try {
        externalTypeHandler0 = new ExternalTypeHandler((ExternalTypeHandler) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler", e);
      }
  }
}
