/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:55:46 GMT 2023
 */

package org.mockito.internal.configuration.injection.filter;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.reflect.Field;
import java.util.LinkedList;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;
import org.mockito.internal.configuration.injection.filter.FinalMockCandidateFilter;
import org.mockito.internal.configuration.injection.filter.MockCandidateFilter;
import org.mockito.internal.configuration.injection.filter.NameBasedCandidateFilter;
import org.mockito.internal.configuration.injection.filter.OngoingInjecter;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NameBasedCandidateFilter_ESTest extends NameBasedCandidateFilter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      NameBasedCandidateFilter nameBasedCandidateFilter0 = new NameBasedCandidateFilter((MockCandidateFilter) null);
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      linkedList0.add((Object) null);
      linkedList0.add((Object) null);
      // Undeclared exception!
      try { 
        nameBasedCandidateFilter0.filterCandidate(linkedList0, (Field) null, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.configuration.injection.filter.NameBasedCandidateFilter", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      FinalMockCandidateFilter finalMockCandidateFilter0 = new FinalMockCandidateFilter();
      NameBasedCandidateFilter nameBasedCandidateFilter0 = new NameBasedCandidateFilter(finalMockCandidateFilter0);
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      OngoingInjecter ongoingInjecter0 = nameBasedCandidateFilter0.filterCandidate(linkedList0, (Field) null, linkedList0);
      assertNotNull(ongoingInjecter0);
  }
}
